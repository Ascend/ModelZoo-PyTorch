#! -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import math
import time
import numbers

import torch
import torch_npu
import deepspeed
import megatron
import pretrain_gpt

from torch import distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from megatron.model.transformer import ParallelAttention, bias_dropout_add_fused_train, \
    bias_dropout_add_fused_inference, get_bias_dropout_add, ParallelMLP
from pretrain_gpt import get_batch_pipe, get_batch, calculate_mos_loss, loss_func
from megatron import print_rank_0, get_args, mpu, get_timers
from megatron.model import GPTModel, GPTModelPipe, DistributedDataParallel as LocalDDP, Float16Module, LayerNorm
from megatron.optimizer import get_megatron_optimizer
from megatron.model.language_model import Embedding, EmbeddingPipe
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model.fused_softmax import FusedScaleMaskSoftmax, ScaledUpperTriangMaskedSoftmax, ScaledMaskedSoftmax
from megatron.model.utils import attention_mask_func, init_method_normal, scaled_init_method_normal, openai_gelu, \
    erf_gelu
from megatron.model.module import float16_to_fp32, fp32_to_float16
from megatron.model.transformer import ParallelTransformerLayerPipe
from megatron.mpu.mappings import copy_to_tensor_model_parallel_region, gather_from_tensor_model_parallel_region, \
    scatter_to_tensor_model_parallel_region, reduce_from_tensor_model_parallel_region
from megatron.training import get_model, get_learning_rate_scheduler
from megatron.utils import unwrap_model
from megatron.checkpointing import load_checkpoint
from megatron.mpu.layers import ColumnParallelLinear, RowParallelLinear
from megatron.model.fused_layer_norm import MixedFusedLayerNorm

from .npu_class import NpuDropout, MatmulApply


def model_provider(pre_process=True, post_process=True):
    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(num_tokentypes=0, parallel_output=True)
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(torch.ones((1, args.seq_length, args.seq_length),
                                                   device=torch.cuda.current_device())) \
                .view(1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half().npu_format_cast(29)
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)
        else:
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    NpuDropout.enable_dropout_ensemble(model)
    return model


def EmbeddingInit(self,
                  hidden_size,
                  vocab_size,
                  max_sequence_length,
                  embedding_dropout_prob,
                  init_method,
                  num_tokentypes=0):
    super(Embedding, self).__init__()

    self.hidden_size = hidden_size
    self.init_method = init_method
    self.num_tokentypes = num_tokentypes

    # Word embeddings (parallel).
    self.word_embeddings = mpu.VocabParallelEmbedding(vocab_size, self.hidden_size, init_method=self.init_method)
    self._word_embeddings_key = 'word_embeddings'

    # Position embedding (serial).
    self.position_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size)
    self._position_embeddings_key = 'position_embeddings'
    # Initialize the position embeddings.
    self.init_method(self.position_embeddings.weight)

    # Token type embedding.
    # Add this as an optional field that can be added through
    # method call so we can load a pretrain model without
    # token types and add them as needed.
    self._tokentype_embeddings_key = 'tokentype_embeddings'
    if self.num_tokentypes > 0:
        self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size)
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)
    else:
        self.tokentype_embeddings = None

    # Embeddings dropout
    self.embedding_dropout = NpuDropout(embedding_dropout_prob)


def ParallelAttentionInit(self, init_method,
                          output_layer_init_method, layer_number,
                          attention_type=AttnType.self_attn,
                          attn_mask_type=AttnMaskType.padding):
    super(ParallelAttention, self).__init__()
    args = get_args()
    self.fp16 = args.fp16
    self.bf16 = args.bf16

    self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
    self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
    if self.apply_query_key_layer_scaling:
        self.attention_softmax_in_fp32 = True
    self.layer_number = max(1, layer_number)
    self.attention_type = None
    self.attn_mask_type = attn_mask_type
    self.num_attention_heads = args.num_attention_heads
    projection_size = args.kv_channels * args.num_attention_heads

    # Per attention head and per partition values.
    world_size = mpu.get_tensor_model_parallel_world_size()
    self.hidden_size_per_partition = mpu.divide(projection_size, world_size)
    self.hidden_size_per_attention_head = mpu.divide(projection_size, args.num_attention_heads)
    self.num_attention_heads_per_partition = mpu.divide(args.num_attention_heads, world_size)

    # Strided linear layer.
    if self.attention_type == AttnType.self_attn:
        self.query_key_value = mpu.ColumnParallelLinear(
            args.hidden_size,
            3 * projection_size,
            gather_output=False,
            init_method=init_method)
        self.query_key_value.weight.data = torch.npu_format_cast(self.query_key_value.weight.data, 29)
    elif self.attention_type == AttnType.cross_attn:
        self.query = mpu.ColumnParallelLinear(
            args.hidden_size,
            projection_size,
            gather_output=False,
            init_method=init_method)

        self.key_value = mpu.ColumnParallelLinear(
            args.hidden_size,
            2 * projection_size,
            gather_output=False,
            init_method=init_method)
    else:
        self.query = mpu.ColumnParallelLinear(
            args.hidden_size,
            projection_size,
            gather_output=False,
            init_method=init_method)
        self.query.weight.data = self.query.weight.data.npu_format_cast(29)
        self.query.bias.data = self.query.bias.data.npu_format_cast(29)

        self.key = mpu.ColumnParallelLinear(
            args.hidden_size,
            projection_size,
            gather_output=False,
            init_method=init_method)
        self.key.weight.data = self.key.weight.data.npu_format_cast(29)
        self.key.bias.data = self.key.bias.data.npu_format_cast(29)

        self.value = mpu.ColumnParallelLinear(
            args.hidden_size,
            projection_size,
            gather_output=False,
            init_method=init_method)
        self.value.weight.data = self.value.weight.data.npu_format_cast(29)
        self.value.bias.data = self.value.bias.data.npu_format_cast(29)

    coeff = None
    self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
    if self.apply_query_key_layer_scaling:
        coeff = self.layer_number
        self.norm_factor *= coeff

    self.scale_mask_softmax = FusedScaleMaskSoftmax(
        self.fp16, self.bf16,
        self.attn_mask_type,
        args.masked_softmax_fusion,
        attention_mask_func,
        self.attention_softmax_in_fp32,
        coeff)

    # Dropout. Note that for a single iteration, this layer will generate
    # different outputs on different number of parallel partitions but
    # on average it should not be partition dependent.

    self.attention_dropout = NpuDropout(args.attention_dropout)

    # Output.
    self.dense = mpu.RowParallelLinear(
        projection_size,
        args.hidden_size,
        input_is_parallel=True,
        init_method=output_layer_init_method,
        skip_bias_add=False)
    self.dense.weight.data = torch.npu_format_cast(self.dense.weight.data, 29)
    self.dense.bias.data = torch.npu_format_cast(self.dense.bias.data, 29)

    if deepspeed.checkpointing.is_configured():
        global get_cuda_rng_tracker, checkpoint
        get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
        checkpoint = deepspeed.checkpointing.checkpoint


def matmul_transpose(tensor1, tensor2):
    return MatmulApply.apply(tensor1, tensor2)


def ParallelAttentionForward(self, hidden_states, attention_mask, layer_past=None, get_key_value=False,
                             encoder_output=None):
    # hidden_states: NZ [b * sq, h]

    # =====================
    # Query, Key, and Value
    # =====================

    batch_size = hidden_states.size()[0] // attention_mask.size()[-1]

    if self.attention_type == AttnType.self_attn:
        # Attention heads NZ: [b * sq, h] --> [b * sq, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # NZ: [b * sq, (np * 3 * hn)] --> [b * sq, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # NZ: [b * sq, np, 3 * hn] --> 3 [b * sq, np, hn]
        (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer.npu_format_cast(2), 3)
    elif self.attention_type == AttnType.cross_attn:
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv_layer, _ = self.key_value(encoder_output)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv_layer.size()[:-1] + (
            self.num_attention_heads_per_partition, 2 * self.hidden_size_per_attention_head)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key_layer, value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query_layer, _ = self.query(hidden_states)
        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query_layer.size()[:-1] + (
            self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        query_layer = query_layer.view(*new_tensor_shape)
    else:
        query_layer, _ = self.query(hidden_states)
        key_layer, _ = self.key(hidden_states)
        value_layer, _ = self.value(hidden_states)

    # ==================================
    # Adjust key and value for inference
    # ==================================

    if layer_past is not None:
        past_key, past_value = layer_past
        key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
        value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)
    if get_key_value:
        present = (key_layer, value_layer)

    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    confusion_transpose = self.hidden_size_per_attention_head % 16 == 0
    if confusion_transpose:
        # NZ: [b * sq, np, hn] -> [b, np, sq, hn]
        query_layer = query_layer.npu_confusion_transpose((0, 2, 1, 3), (batch_size, query_layer.size(0) // batch_size,
                                                                         self.num_attention_heads_per_partition,
                                                                         self.hidden_size_per_attention_head),
                                                          False)
        # # NZ: [b * sq, np, hn] -> [b, np, hn, sq]
        key_layer = key_layer.npu_confusion_transpose((0, 2, 1, 3), (batch_size, key_layer.size(0) // batch_size,
                                                                     self.num_attention_heads_per_partition,
                                                                     self.hidden_size_per_attention_head),
                                                      False)
        matmul_result = matmul_transpose(query_layer, key_layer)
    else:
        query_layer = query_layer.reshape(batch_size, query_layer.size(0) // batch_size,
                                          self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        query_layer = query_layer.transpose(1, 2)

        key_layer = key_layer.reshape(batch_size, key_layer.size(0) // batch_size,
                                      self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        key_layer = key_layer.permute(0, 2, 3, 1)
        matmul_result = torch.matmul(query_layer, key_layer)

    # ==================================================
    # Update attention mask for inference. [b, np, sq, sk]
    # ==================================================

    if get_key_value:
        with torch.no_grad():
            if layer_past is not None:
                attention_mask = attention_mask[..., attention_scores.size(3) - 1, :attention_scores.size(3)].unsqueeze(
                    2)
            else:
                attention_mask = attention_mask[..., :attention_scores.size(3), :attention_scores.size(3)]

    # ===========================
    # Attention probs and dropout
    # ===========================

    # attention scores and attention mask [b, np, sq, sk]
    attention_probs = self.scale_mask_softmax(matmul_result, attention_mask, self.norm_factor)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    with mpu.get_cuda_rng_tracker().fork():
        attention_probs = self.attention_dropout(attention_probs)

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    if confusion_transpose:
        # NZ: [b * sk, np, hn] -> [b, np, sk, hn]
        value_layer = value_layer.npu_confusion_transpose((0, 2, 1, 3), (batch_size, value_layer.size(0) // batch_size,
                                                                         self.num_attention_heads_per_partition,
                                                                         self.hidden_size_per_attention_head),
                                                          False)
    else:
        value_layer = value_layer.reshape(batch_size, value_layer.size(0) // batch_size,
                                          self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        value_layer = value_layer.transpose(1, 2)

    # matmul: [b * np, sq, hn]
    context_layer = torch.matmul(attention_probs, value_layer)  # [b, np, sk, hn]

    # [b * sq, h]
    if confusion_transpose:
        context_layer = context_layer.npu_confusion_transpose((0, 2, 1, 3),
                                                              (context_layer.size(0) * context_layer.size(2),
                                                               context_layer.size(1) * context_layer.size(3)),
                                                              True)
    else:
        context_layer = context_layer.transpose(1, 2)
        context_layer = context_layer.reshape(context_layer.size(0) * context_layer.size(1), -1)

    output, bias = self.dense(context_layer)

    if get_key_value:
        output = [output, present]

    return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    x = x + bias if bias is not None else x
    out = NpuDropout.dropout_functional(x, p=prob, training=training)
    out = residual + out
    return out


def ParallelMLPInit(self, init_method, output_layer_init_method, moe=False):
    super(ParallelMLP, self).__init__()
    args = get_args()

    # Project to 4h.
    self.dense_h_to_4h = mpu.ColumnParallelLinear(
        args.hidden_size,
        args.ffn_hidden_size,
        gather_output=False,
        init_method=init_method,
        skip_bias_add=False,
        moe=False)
    self.dense_h_to_4h.weight.data = torch.npu_format_cast(self.dense_h_to_4h.weight.data, 29)
    self.dense_h_to_4h.bias.data = torch.npu_format_cast(self.dense_h_to_4h.bias.data, 29)

    self.bias_gelu_fusion = args.bias_gelu_fusion
    self.activation_func = torch.nn.functional.gelu
    if args.openai_gelu:
        self.activation_func = openai_gelu
    elif args.onnx_safe:
        self.activation_func = erf_gelu

    # Project back to h.
    self.dense_4h_to_h = mpu.RowParallelLinear(
        args.ffn_hidden_size,
        args.hidden_size,
        input_is_parallel=True,
        init_method=output_layer_init_method,
        skip_bias_add=False,
        moe=False)
    self.dense_4h_to_h.weight.data = torch.npu_format_cast(self.dense_4h_to_h.weight.data, 29)
    self.dense_4h_to_h.bias.data = torch.npu_format_cast(self.dense_4h_to_h.bias.data, 29)


def ParallelMLPForward(self, hidden_states):
    # [s, b, 4hp]
    intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
    intermediate_parallel = intermediate_parallel + bias_parallel if bias_parallel is not None else intermediate_parallel
    if self.bias_gelu_fusion:
        intermediate_parallel = torch.fast_gelu(intermediate_parallel)
    else:
        intermediate_parallel = self.activation_func(intermediate_parallel)

    # [s, b, h]
    output, output_bias = self.dense_4h_to_h(intermediate_parallel)
    return output, output_bias


def EmbeddingForward(self, input_ids, position_ids, tokentype_ids=None):
    # Embeddings.
    words_embeddings = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids)
    embeddings = words_embeddings + position_embeddings
    embeddings = embeddings.view(-1, embeddings.size()[-1]).clone()
    if tokentype_ids is not None:
        assert self.tokentype_embeddings is not None
        embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
    else:
        assert self.tokentype_embeddings is None

    # Dropout.
    embeddings = self.embedding_dropout(embeddings)

    return embeddings


def ParallelTransformerLayerForward(self, hidden_states, attention_mask, encoder_output=None, enc_dec_attn_mask=None,
                                    layer_past=None, get_key_value=False):
    # hidden_states: [b, s, h]

    # Layer norm at the beginning of the transformer layer.
    hidden_states = hidden_states.npu_format_cast(29)
    layernorm_output = self.input_layernorm(hidden_states)
    # Self attention.
    attention_output, attention_bias = self.attention(layernorm_output, attention_mask,
                                                      layer_past=layer_past, get_key_value=get_key_value)

    if get_key_value:
        attention_output, presents = attention_output

    # Residual connection.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = hidden_states

    # jit scripting for a nn.module (with dropout) is not
    # trigerring the fusion kernel. For now, we use two
    # different nn.functional routines to account for varying
    # dropout semantics during training and inference phases.
    if self.bias_dropout_fusion:
        if self.training:
            bias_dropout_add_func = bias_dropout_add_fused_train
        else:
            bias_dropout_add_func = bias_dropout_add_fused_inference
    else:
        bias_dropout_add_func = get_bias_dropout_add(self.training)

    # re-enable torch grad to enable fused optimization.
    with torch.enable_grad():
        layernorm_input = bias_dropout_add_func(attention_output,
                                                attention_bias,
                                                residual,
                                                self.hidden_dropout)

    # Layer norm post the self attention.
    layernorm_output = self.post_attention_layernorm(layernorm_input)

    if self.layer_type == LayerType.decoder:
        attention_output, attention_bias = self.inter_attention(layernorm_output, enc_dec_attn_mask,
                                                                encoder_output=encoder_output)
        # residual connection
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(attention_output,
                                                    attention_bias,
                                                    residual,
                                                    self.hidden_dropout)

        # Layer norm post the decoder attention
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

    # MLP.
    if self.num_experts == 1:
        mlp_output, mlp_bias = self.mlp(layernorm_output)
    else:
        mlp_output, moe_loss, _ = self.mlp(layernorm_output)

    # Second residual connection.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = layernorm_input

    # re-enable torch grad to enable fused optimization.
    with torch.enable_grad():
        if self.num_experts <= 1:
            output = bias_dropout_add_func(mlp_output, mlp_bias, residual, self.hidden_dropout)
        else:
            output = mlp_output + residual

    if get_key_value:
        output = [output, presents]

    return output, None


def ColumnParallelLinearForward(self, input_):
    # Set up backprop all-reduce.
    input_parallel = copy_to_tensor_model_parallel_region(input_)
    # Matrix multiply.

    bias = self.bias if not self.skip_bias_add else None
    input_shape = input_parallel.size()
    if input_parallel.dim() == 3:
        input_parallel = input_parallel.view(-1, input_shape[2])
        output_parallel = torch.npu_linear(input_parallel, self.weight, bias).view(input_shape[0], input_shape[1], -1)
    elif input_parallel.dim() == 2:
        output_parallel = torch.npu_linear(input_parallel, self.weight, bias)
    else:
        output_parallel = torch.nn.functional.linear(input_parallel, self.weight, bias)

    if self.gather_output:
        # All-gather across the partitions.
        output = gather_from_tensor_model_parallel_region(output_parallel)
    else:
        output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def RowParallelLinearForward(self, input_):
    # Set up backprop all-reduce.
    bias = self.bias if not self.skip_bias_add else None
    if self.input_is_parallel:
        input_parallel = input_
    else:
        input_parallel = scatter_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    input_shape = input_parallel.size()
    if input_parallel.dim() == 3:
        input_parallel = input_parallel.view(-1, input_shape[2])
        output_parallel = torch.npu_linear(input_parallel, self.weight, bias).view(input_shape[0], input_shape[1], -1)
    elif input_parallel.dim() == 2:
        output_parallel = torch.npu_linear(input_parallel, self.weight, bias)
    else:
        output_parallel = torch.nn.functional.linear(input_parallel, self.weight, bias)
    # All-reduce across all the partitions.
    output_ = reduce_from_tensor_model_parallel_region(output_parallel)
    if not self.skip_bias_add:
        output = output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias


def setup_model_and_optimizer(model_provider_func, teacher=False):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func)

    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))

    if args.inference:
        optimizer = None
        lr_scheduler = None
    else:
        if teacher:
            optimizer = None
        else:
            optimizer = get_megatron_optimizer(unwrapped_model)
        lr_scheduler = get_learning_rate_scheduler(optimizer)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        pp = mpu.get_pipeline_model_parallel_world_size()
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model[0],
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu if args.no_pipeline_parallel else None
        )
        if isinstance(model, deepspeed.PipelineEngine):
            # hack to get batch_fn from pretrain_gpt.py
            model.set_batch_fn(model.module._megatron_batch_fn)

            assert model.grid.get_pipe_parallel_rank() == mpu.get_pipeline_model_parallel_rank()
            assert model.grid.get_slice_parallel_rank() == mpu.get_tensor_model_parallel_rank()
            assert model.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank()

        model = [model]
    if args.load is not None:
        timers = get_timers()
        # Extra barrier is added to make sure all ranks report the
        # max time.
        torch.distributed.barrier()
        timers('load-checkpoint').start()
        if args.mos:
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler, strict=False, load_only_weights=False)
        else:
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
        torch.distributed.barrier()
        timers('load-checkpoint').stop()
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
            and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()
    return model, optimizer, lr_scheduler


def GPTModelPipeInit(self,
                     num_tokentypes=0,
                     parallel_output=True):
    args = get_args()
    self.parallel_output = parallel_output

    init_method = init_method_normal(args.init_method_std)

    self.specs = []

    def _to_float16(inputs):
        if args.fp16:
            return fp32_to_float16(inputs, lambda v: v.half())
        elif args.bf16:
            return fp32_to_float16(inputs, lambda v: v.bfloat16())
        else:
            return inputs

    self.specs.append(_to_float16)

    # Embedding layer
    self.specs.append(TiedLayerSpec('embed',
                                    EmbeddingPipe,
                                    args.hidden_size,
                                    args.padded_vocab_size,
                                    args.max_position_embeddings,
                                    args.hidden_dropout,
                                    init_method=init_method,
                                    num_tokentypes=num_tokentypes,
                                    tied_weight_attr='word_embeddings_weight'))

    for layer_idx in range(args.num_layers):
        self.specs.append(
            LayerSpec(ParallelTransformerLayerPipe,
                      init_method=init_method,
                      output_layer_init_method=scaled_init_method_normal(args.init_method_std,
                                                                         args.num_layers),
                      layer_number=layer_idx,
                      self_attn_mask_type=AttnMaskType.causal))

    # Final layernorm after transformer layers
    self.specs.append(
        LayerSpec(LayerNorm,
                  args.hidden_size,
                  eps=args.layernorm_epsilon))

    def _logits_helper(embedding, lm_output):
        """A wrapper to massage inputs/outputs from pipeline. """
        return parallel_lm_logits(
            lm_output,
            embedding.word_embeddings_weight,
            self.parallel_output)

    self.specs.append(
        TiedLayerSpec('embed',
                      EmbeddingPipe,
                      args.hidden_size,
                      args.padded_vocab_size,
                      args.max_position_embeddings,
                      args.hidden_dropout,
                      init_method=init_method,
                      num_tokentypes=num_tokentypes,
                      forward_fn=_logits_helper,
                      tied_weight_attr='word_embeddings_weight')
    )

    # Convert to fp32 if needed
    if args.fp16 or args.bf16:
        self.specs.append(float16_to_fp32)

    if args.checkpoint_activations:
        interval = args.checkpoint_num_layers
    else:
        interval = 0

    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
    topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                         num_mp=mpu.get_tensor_model_parallel_world_size(),
                                         num_dp=mpu.get_data_parallel_world_size())

    super(GPTModelPipe, self).__init__(layers=self.specs, loss_fn=CrossEntropy, topology=topo,
                                       activation_checkpoint_interval=interval, partition_method='type:transformer')


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    if bias is None:
        logits_parallel = torch.nn.functional.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = torch.nn.functional.linear(input_parallel, word_embeddings_weight, bias)
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_tensor_model_parallel_region(logits_parallel)


def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]
    args = get_args()
    if args.tensor_model_parallel_size > 1:
        # [b * sq, h] --> [b, sq, h]
        output = output.view(labels.size(0), labels.size(1), -1)

        losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
        loss_mask = loss_mask.view(-1)
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    else:
        labels = labels.flatten()
        loss_mask = loss_mask.flatten()
        if loss_mask.sum() == loss_mask.numel():
            loss = torch.nn.functional.cross_entropy(output.contiguous().float(), labels)
        else:
            losses = torch.nn.functional.cross_entropy(output.contiguous().float(), labels, reduction='none')
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


def FusedScaleMaskSoftmaxInit(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
):
    super(FusedScaleMaskSoftmax, self).__init__()
    self.input_in_fp16 = input_in_fp16
    self.input_in_bf16 = input_in_bf16
    assert not (
            self.input_in_fp16 and self.input_in_bf16
    ), "both fp16 and bf16 flags cannot be active at the same time."
    self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
    self.attn_mask_type = attn_mask_type
    self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
    self.mask_func = mask_func
    self.softmax_in_fp32 = softmax_in_fp32
    self.scale = scale
    self.mask_tri = None

    assert (
            self.scale is None or softmax_in_fp32
    ), "softmax should be in fp32 when scaled"


def FusedScaleMaskSoftmaxForward(self, input, mask, norm_factor):
    # [b, np, sq, sk]
    assert input.dim() == 4
    data_size = input.size()
    query_seq_len = data_size[-2]
    key_seq_len = data_size[-1]
    attn_batch_size = data_size[0] * data_size[1]

    # constraints on various tensor dimensions to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = key_seq_len > 16 and key_seq_len <= 2048 and \
                               query_seq_len % 4 == 0 and attn_batch_size % 4 == 0

    # invoke custom kernel
    if self.input_in_float16 and mask is not None and \
            custom_kernel_constraint and self.scaled_masked_softmax_fusion:
        scale = self.scale if self.scale is not None else 1.0

        if self.attn_mask_type == AttnMaskType.causal:
            assert query_seq_len == key_seq_len, \
                "causal mask is only for self attention"
            if self.mask_tri is None:
                self.mask_tri = torch.triu(
                    torch.ones((1, 1, input.shape[2], input.shape[3]), dtype=input.dtype, device=input.device),
                    diagonal=1).npu_format_cast(29).bool()
            probs = torch.npu_scaled_masked_softmax(input, self.mask_tri, self.scale * (1.0 / norm_factor), False)
            probs = probs.half()
        else:
            assert self.attn_mask_type == AttnMaskType.padding
            probs = ScaledMaskedSoftmax.apply(input, mask, scale)
    else:
        probs = torch_npu.npu_scaled_masked_softmax(input, mask, self.scale * (1.0 / norm_factor), False)

    return probs


def MixedFusedLayerNormInit(self, normalized_shape, eps=1e-5):
    super(MixedFusedLayerNorm, self).__init__()

    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)
    self.normalized_shape = torch.Size(normalized_shape)
    self.eps = eps
    self.weight = torch.nn.Parameter(torch.Tensor(*normalized_shape))
    self.bias = torch.nn.Parameter(torch.Tensor(*normalized_shape))
    self.reset_parameters()


def MixedFusedLayerNormForward(self, input):
    return torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


from deepspeed.runtime import engine
from deepspeed.utils import logger, log_dist
from deepspeed_npu.adaptor_ops_adam_fused_adam import FusedAdamNPU
from deepspeed.runtime.config import LAMB_OPTIMIZER, ONEBIT_ADAM_OPTIMIZER
from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer


def _configure_fp16_optimizer(self, optimizer):
    initial_dynamic_scale = self.initial_dynamic_scale()
    dynamic_loss_args = self.dynamic_loss_scale_args()
    clip_grad = self.gradient_clipping()
    fused_opts = (FusedAdamNPU)

    if isinstance(optimizer, fused_opts) \
            or self.optimizer_name() == ONEBIT_ADAM_OPTIMIZER:
        if self.dynamic_loss_scale():
            log_dist("Creating fp16 optimizer with dynamic loss scale", ranks=[0])
            timers = self.timers if self.wall_clock_breakdown() else None
            optimizer = FP16_Optimizer(
                optimizer,
                deepspeed=self,
                dynamic_loss_scale=True,
                initial_dynamic_scale=initial_dynamic_scale,
                dynamic_loss_args=dynamic_loss_args,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_adam_legacy=self.optimizer_legacy_fusion(),
                timers=timers,
            )
        else:
            log_dist(
                "Creating fp16 optimizer with static loss scale: {}".format(
                    self.loss_scale()),
                ranks=[0],
            )
            optimizer = FP16_Optimizer(
                optimizer,
                deepspeed=self,
                static_loss_scale=self.loss_scale(),
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_adam_legacy=self.optimizer_legacy_fusion(),
            )
    else:
        log_dist("Creating fp16 unfused optimizer with dynamic loss scale",
                 ranks=[0])
        optimizer = FP16_UnfusedOptimizer(
            optimizer,
            deepspeed=self,
            static_loss_scale=self.loss_scale(),
            dynamic_loss_scale=self.dynamic_loss_scale(),
            dynamic_loss_args=dynamic_loss_args,
            mpu=self.mpu,
            clip_grad=clip_grad,
            fused_lamb_legacy=self.optimizer_name() == LAMB_OPTIMIZER,
        )

    return optimizer


def _compile_dependencies():
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.data.dataset_utils import compile_helper
        compile_helper()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)


@classmethod
def FromMeta(cls, meta, local_part, group, device='cuda'):
    # Fixbug for NPU 
    meta = meta.long()  
    assert meta.dtype == torch.long
    dummy = torch.ones(dist.get_world_size(group=group))
    part_obj = cls(tensor=dummy, group=group)

    meta = meta.tolist()

    # [N, list0, ..., listN-1]
    part_obj.orig_size = meta[1:(1 + meta[0])]
    meta = meta[1 + meta[0]:]

    part_obj.orig_device = device
    part_obj.local_data = local_part.detach()

    part_obj.group = group

    # Partition is encoded like the rowptr of a CSR matrix:
    # [num_parts, rank, 0, part_1, ..., part_num_parts]
    # TODO: support shuffle between different partition granularities
    assert part_obj.num_parts == meta[0]
    assert part_obj.rank == meta[1]
    part_obj.partition = meta[2:]  # length num_parts+1

    return part_obj


# set npu option
option = {
    "ACL_OP_COMPILER_CACHE_MODE": "enable",
    "ACL_OP_COMPILER_CACHE_DIR": "./cache",
    "MM_BMM_ND_ENABLE": "disable",
    "ACL_OP_SELECT_IMPL_MODE": "high_performance",
    "ACL_OPTYPELIST_FOR_IMPLMODE": "LayerNorm",
}

print("option:", option)
torch.npu.set_option(option)

# fix mismatch for NPU in deepspeed
engine.DeepSpeedEngine._configure_fp16_optimizer = _configure_fp16_optimizer
deepspeed.runtime.utils.PartitionedTensor.from_meta = FromMeta

# replace with NPU dropout
pretrain_gpt.model_provider = model_provider
megatron.model.language_model.Embedding.__init__ = EmbeddingInit
megatron.model.transformer.ParallelAttention.__init__ = ParallelAttentionInit
megatron.model.transformer.bias_dropout_add = bias_dropout_add

# replace baddbmm with bmm+mul and qkv split
megatron.model.transformer.ParallelAttention.forward = ParallelAttentionForward

# remove moe loss
megatron.model.transformer.ParallelTransformerLayer.forward = ParallelTransformerLayerForward

# replace with fast gelu
megatron.model.transformer.ParallelMLP.forward = ParallelMLPForward

# fused softmax
megatron.model.fused_softmax.FusedScaleMaskSoftmax.__init__ = FusedScaleMaskSoftmaxInit
megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward = FusedScaleMaskSoftmaxForward

# set NZ format
megatron.model.transformer.ParallelMLP.__init__ = ParallelMLPInit
megatron.training.setup_model_and_optimizer = setup_model_and_optimizer
megatron.model.gpt_model.GPTModelPipe.__init__ = GPTModelPipeInit
megatron.model.language_model.Embedding.forward = EmbeddingForward
megatron.model.language_model.parallel_lm_logits = parallel_lm_logits
megatron.mpu.layers.ColumnParallelLinear.forward = ColumnParallelLinearForward
megatron.mpu.layers.RowParallelLinear.forward = RowParallelLinearForward
megatron.model.gpt_model.CrossEntropy = CrossEntropy
megatron.model.fused_layer_norm.MixedFusedLayerNorm.__init__ = MixedFusedLayerNormInit
megatron.model.fused_layer_norm.MixedFusedLayerNorm.forward = MixedFusedLayerNormForward
megatron.initialize._compile_dependencies = _compile_dependencies
