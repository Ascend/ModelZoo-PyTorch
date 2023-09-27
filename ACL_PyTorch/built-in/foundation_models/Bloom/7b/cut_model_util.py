import os
import time
import argparse
import torch
import torch_npu
from transformers import AutoTokenizer, BloomForCausalLM
from transformers.models.bloom.configuration_bloom import BloomConfig
from torch_npu.contrib import transfer_to_npu


torch.npu.set_device("npu:2")


# cut weights
# cut_row_keys :dim 0  cut_col_keys :dim 1  nn.linear: x*A.T
def cut_weights(model_cut, world_size_cut, cut_row_keys, cut_col_keys):
    state_dict_list_ = [{} for _ in range(world_size_cut)]
    for key, tensor in model_cut.state_dict().items():
        key_short = key.split('.')[-2]
        key_type = key.split('.')[-1]

        if key_short == 'query_key_value':
            num_heads = 32
            head_dim = 128
            if key_type == "weight":
                tensor = tensor.view(num_heads, 3, head_dim, num_heads * head_dim)
            elif key_type == "bias":
                tensor = tensor.view(num_heads, 3, head_dim)
            tensor_list = (tensor[:, 0, ...], tensor[:, 1, ...], tensor[:, 2, ...])
            cut_tensor_list = [torch.Tensor([]), torch.Tensor([])]
            for i in range(3):
                cut_tensor_list[0] = torch.cat(
                    (cut_tensor_list[0], torch.chunk(tensor_list[i], world_size_cut, dim=0)[0]), 1)
                cut_tensor_list[1] = torch.cat(
                    (cut_tensor_list[1], torch.chunk(tensor_list[i], world_size_cut, dim=0)[1]), 1)
            if key_type == "weight":
                cut_tensor_list[0] =  cut_tensor_list[0].reshape(num_heads * head_dim * 3 // 2, num_heads * head_dim)
                cut_tensor_list[1] =  cut_tensor_list[1].reshape(num_heads * head_dim * 3 // 2, num_heads * head_dim)
            elif key_type == "bias":
                cut_tensor_list[0] =  cut_tensor_list[0].reshape(num_heads * head_dim * 3 // 2)
                cut_tensor_list[1] =  cut_tensor_list[1].reshape(num_heads * head_dim * 3 // 2)
        else:
            if key_short in cut_row_keys:
                cut_tensor_list = torch.chunk(tensor, world_size_cut, dim=0)
            elif key_short in cut_col_keys:
                if key_type == "weight":
                    cut_tensor_list = torch.chunk(tensor, world_size_cut, dim=1)
                elif key_type == "bias":
                    tensor = tensor / 2
                    cut_tensor_list = [tensor] * world_size_cut
            else:
                cut_tensor_list = [tensor] * world_size_cut

        for j in range(world_size_cut):
            state_dict_list_[j][key] = cut_tensor_list[j]
    return state_dict_list_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default="./model/",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='./',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default=2,
        help="world_size",
    )
    parser.add_argument(
        "--cut_row_keys",
        default=['dense_h_to_4h'],
        help="cut_row_keys",
    )
    parser.add_argument(
        "--cut_col_keys",
        default=['dense', 'dense_4h_to_h'],
        help="cut_col_keys",
    )
    args = parser.parse_args()
    args.world_size = int(args.world_size)
    tokenizer = AutoTokenizer.from_pretrained(
        args.input_path, use_fast=False)  # load the tokenizer
    tokenizer.save_pretrained(
        args.output_path + '/tokenizer')  # save the tokenizer

    model = BloomForCausalLM.from_pretrained(
        args.input_path, torch_dtype=torch.float16)  # load the raw model
    state_dict_list = cut_weights(
        model, args.world_size, args.cut_row_keys, args.cut_col_keys)  # cut the weight
    model_config = model.config
    # create new model config, add the world size parameter, 
    # the model size will be cut according to the world size in the model file
    create_config = BloomConfig(
        apply_residual_connection_post_layernorm=model_config.apply_residual_connection_post_layernorm,
        architectures=model_config.architectures,
        attention_dropout=model_config.attention_dropout,
        attention_softmax_in_fp32=model_config.attention_softmax_in_fp32,
        bias_dropout_fusion=model_config.bias_dropout_fusion,
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,
        hidden_dropout=model_config.hidden_dropout,
        hidden_size=model_config.hidden_size,
        initializer_range=model_config.initializer_range,
        layer_norm_epsilon=model_config.layer_norm_epsilon,
        masked_softmax_fusion=model_config.masked_softmax_fusion,
        model_type=model_config.model_type,
        n_head=model_config.n_head,
        n_inner=model_config.n_inner,
        n_layer=model_config.n_layer,
        offset_alibi=model_config.offset_alibi,
        pad_token_id=model_config.pad_token_id,
        pretraining_tp=model_config.pretraining_tp,
        skip_bias_add=model_config.skip_bias_add,
        skip_bias_add_qkv=model_config.skip_bias_add_qkv,
        slow_but_exact=model_config.slow_but_exact,
        torch_dtype=model_config.torch_dtype,
        transformers_version="4.25.1",
        unk_token_id=model_config.unk_token_id,
        use_cache=model_config.use_cache,
        vocab_size=model_config.vocab_size,
        world_size=args.world_size,
    )
    # create new model according to the model config
    create_model = BloomForCausalLM(create_config)
    for k in range(args.world_size):
        # load the weights to the model
        create_model.load_state_dict(state_dict_list[k])
        create_model.save_pretrained(
            args.output_path + '/part_model/' + str(k) + '/')  # save model
    print('save succcessfully')
