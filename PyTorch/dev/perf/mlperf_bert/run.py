# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import random

import numpy as np
import torch
import torch_npu
import torch.multiprocessing
from torch_npu.contrib import transfer_to_npu

from apex import amp
from optim import Lamb
from modeling import BertForPretraining, BertConfig
from schedulers import LinearWarmupPolyDecayScheduler
import mlperf_logger
import utils
import run_pretraining
from run_pretraining import found_resume_checkpoint, global_batch_size

torch.multiprocessing.set_sharing_strategy('file_system')


class NPUWorkInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, idx):
        np.random.seed(seed=self.seed + idx)
        random.seed(self.seed + idx)
        torch.multiprocessing.set_sharing_strategy('file_system')


def prepare_model_and_optimizer(args, device):
    global_step = 0
    args.resume_step = 0
    checkpoint = None

    config = BertConfig.from_json_file(args.bert_config_path)
    config.fused_mha = args.fused_mha
    config.fused_gelu_bias = args.fused_gelu_bias
    config.dense_seq_output = args.dense_seq_output
    config.unpad = args.unpad
    config.pad = args.pad
    config.fuse_qkv = not args.disable_fuse_qkv
    config.fuse_scale = not args.disable_fuse_scale
    config.fuse_mask = not args.disable_fuse_mask
    config.fuse_dropout = args.enable_fuse_dropout
    config.apex_softmax = not args.disable_apex_softmax
    config.enable_stream = args.enable_stream
    config.hidden_dropout_prob = 0 # TODO: 临时规避dropout导致精度异常，根因需要进一步定位
    config.attention_probs_dropout_prob = 0
    if config.fuse_mask == True:
        config.apex_softmax = True
    if config.pad == False:
        config.enable_stream = True
    if config.unpad == True:
        config.fused_mha = False

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    # Load from Pyt checkpoint - either given as init_checkpoint, or picked up from output_dir if found
    if args.init_checkpoint is not None or found_resume_checkpoint(args):
        # Prepare model
        model = BertForPretraining(config)
        if args.init_checkpoint is None:  # finding checkpoint in output_dir
            checkpoint_str = "phase2_ckpt_*.pt" if args.phase2 else "phase1_ckpt_*.pt"
            model_names = [f for f in glob.glob(os.path.join(args.output_dir, checkpoint_str))]
            global_step = max([int(x.split('.pt')[0].split('_')[-1].strip()) for x in model_names])
            args.resume_step = global_step  # used for throughput computation

            resume_init_checkpoint = os.path.join(args.output_dir, checkpoint_str.replace("*", str(global_step)))
            print(
                "Setting init checkpoint to %s - which is the latest in %s" % (resume_init_checkpoint, args.output_dir))
            checkpoint = torch.load(resume_init_checkpoint, map_location="cpu")
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location="cpu")["model"]

        # Fused MHA requires a remapping of checkpoint parameters
        if config.fused_mha:
            checkpoint_remapped = remap_attn_parameters(checkpoint)
            model.load_state_dict(checkpoint_remapped, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=True)
    else:  # Load from TF Checkpoint
        model = BertForPretraining.from_pretrained(args.init_tf_checkpoint, from_tf=True, config=config)

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    mlperf_logger.log_event(key=mlperf_logger.constants.OPT_BASE_LR,
                            value=args.learning_rate, sync=False)

    # Set max_grad_norm=65536. to avoid clip after allreduce
    optimizer = Lamb(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2),
                          max_grad_norm=65536.0)
    if os.getenv('ALLOW_FP32'):
        optimizer = torch_npu.optim.NpuFusedLamb(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2),
                          max_grad_norm=65536.0)
    mlperf_logger.log_event(key='optimizer', value=optimizer.__class__.__name__, sync=False)

    mlperf_logger.log_event(key='opt_epsilon', value=optimizer.defaults['eps'],
                            sync=False)
    b1, b2 = optimizer.defaults['betas']
    mlperf_logger.log_event(key='opt_lamb_beta_1', value=b1, sync=False)
    mlperf_logger.log_event(key='opt_lamb_beta_2', value=b2, sync=False)
    mlperf_logger.log_event(key='opt_lamb_weight_decay_rate',
                            value=optimizer.defaults['weight_decay'],
                            sync=False)

    if args.warmup_steps == 0:
        warmup_steps = int(args.max_steps * args.warmup_proportion)
        warmup_start = 0
    else:
        warmup_steps = args.warmup_steps
        warmup_start = args.start_warmup_step
    lr_scheduler = LinearWarmupPolyDecayScheduler(optimizer, start_warmup_steps=warmup_start, warmup_steps=warmup_steps,
                                                  total_steps=args.max_steps, end_learning_rate=0.0, degree=1.0)

    if args.fp16 and not os.getenv('ALLOW_FP32'):
        if args.loss_scale == 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic",
                                              master_weights=True)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=args.loss_scale,
                                              master_weights=True)
        amp._amp_state.loss_scalers[0]._loss_scale = float(os.getenv("INIT_LOSS_SCALE", 2 ** 20))

    if found_resume_checkpoint(args):
        # restores m,v states (only if resuming checkpoint, not for init_checkpoint and init_tf_checkpoint for now)
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Restore AMP master parameters
        if args.fp16 and not os.getenv('ALLOW_FP32'):
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint['master params']):
                param.data.copy_(saved_param.data)

    if args.local_rank != -1:
        if not args.allreduce_post_accumulation:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                bucket_cap_mb=8192,
            )
        else:
            flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,))

    return model, optimizer, lr_scheduler, checkpoint, global_step


run_pretraining.WorkerInitObj = NPUWorkInitObj
run_pretraining.prepare_model_and_optimizer = prepare_model_and_optimizer

def main():
    now = time.time()
    args, final_loss, train_time_raw = run_pretraining.main()

    if utils.is_main_process():
        e2e_time = time.time() - now
        training_perf = global_batch_size(args) \
                        * (args.max_steps - args.resume_step + run_pretraining.skipped_steps) / train_time_raw
        if args.do_train:
            print({"e2e_time": e2e_time,
                   "training_sequences_per_second": training_perf,
                   "final_loss": final_loss,
                   "raw_train_time": train_time_raw})
        else:
            print({"e2e_time": e2e_time})


if __name__ == "__main__":
    torch_npu.npu.set_compile_mode(jit_compile=False)
    if os.getenv('ALLOW_FP32') or os.getenv('ALLOW_HF32'):
        torch.npu.config.allow_internal_format = False
        if os.getenv('ALLOW_FP32'):
            torch.npu.conv.allow_hf32 = False
            torch.npu.matmul.allow_hf32 = False
    main()

