# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
import sys
import config
import argparse

import torch
import torch_aie
from torch_aie import _enums

import lib.models.crnn as crnn

COSINE_THRESHOLD = 0.999


def cosine_similarity(gt_tensor, pred_tensor):
    gt_tensor = gt_tensor.flatten().to(torch.float32)
    pred_tensor = pred_tensor.flatten().to(torch.float32)
    if torch.sum(gt_tensor) == 0.0 or torch.sum(pred_tensor) == 0.0:
        if torch.allclose(gt_tensor, pred_tensor, atol=1e-4, rtol=1e-4, equal_nan=True):
            return 1.0
    res = torch.nn.functional.cosine_similarity(gt_tensor, pred_tensor, dim=0, eps=1e-6)
    res = res.cpu().detach().item()

    return res


def __load_checkpoint(model, checkpoint_filepath, device):
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)


def trace_compile(torch_model, args):
    input_shape = (args.batch_size, 1, 32, 160)
    inputs = torch.randn(size = input_shape, dtype = torch.float32)
    input_data = [ inputs ]

    # trace model
    print("trace start. ")
    traced_model = torch.jit.trace(torch_model, input_data)
    print("trace done. ")
    # print("traced model is ", traced_model.graph)

    traced_model.eval()
    jit_result = traced_model(inputs)
    print("torch_aie compile start !")
    torch_aie.set_device(0)
    compile_inputs = [torch_aie.Input(shape = input_shape, dtype = torch.float32, format = torch_aie.TensorFormat.NCHW)]
    compiled_model = torch_aie.compile(
        traced_model,
        inputs = compile_inputs,
        precision_policy = _enums.PrecisionPolicy.FP16,
        soc_version = "Ascend310P3",
        optimization_level = 0
    )
    print("torch_aie compile done !")
    aie_result = compiled_model(inputs.to("npu"))
    print("compiled model is ", compiled_model.graph)
    compiled_model.save(args.pt_dir)
    print("torch aie compiled model saved. ")

    com_res = True
    res = cosine_similarity(jit_result, aie_result.to("cpu"))
    print(res)
    if res < COSINE_THRESHOLD:
            com_res = False
    if com_res:
        print("Compare success ! NPU model have the same output with CPU model !")
    else:
        print("Compare failed ! Outputs of NPU model are not the same with CPU model !")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_filepath",
                        default="./output/checkpoints/mixed_second_finetune_acc_97P7.pth",
                        type=str,
                        help="The original torch pt file from pretraining")   
    parser.add_argument("--save_dir",
                        default="./",
                        type=str,
                        help="The path of the directory that stores the compiled model")   
    parser.add_argument("--config_file",
                        default="./lib/config/360CC_config.yaml",
                        type=str,
                        help="The crnn model config")
    parser.add_argument('--batch_size',
                        default=8,
                        type=int,
                        help="batch size")
    args = parser.parse_args()

    config_filepath = args.config_file
    checkpoint_filepath = args.checkpoint_filepath
    args.pt_dir = args.save_dir + 'crnn_sierkinhane_bs{}.pt'.format(args.batch_size)

    torch_model = crnn.get_crnn(config.get_config(config_filepath)).to("cpu")
    __load_checkpoint(torch_model, checkpoint_filepath, "cpu")
    torch_model.eval()

    trace_compile(torch_model, args)