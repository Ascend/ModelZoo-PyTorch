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
import argparse
import torch
import torch_aie
from torch_aie import _enums
import parse


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


def get_torch_model(ar):
    _, model, _ = parse.load_data_model(ar)
    return model


def aie_compile(torch_model, ar):
    torch_model.eval()
    input_shape = (ar.batch_size, ar.max_seq_length)
    input_ids = torch.randint(high = 1, size = input_shape, dtype = torch.int32)
    att_mask = torch.randint(high = 3, size = input_shape, dtype = torch.int32)
    token_ids = torch.randint(high = 1, size = input_shape, dtype = torch.int32)
    input_data = [ input_ids, att_mask, token_ids]
    print("start to trace model.")
    traced_model = torch.jit.trace(torch_model, input_data)
    traced_model.eval()
    print("trace done !")

    print("torch_aie compile start.")
    torch_aie.set_device(0)
    compile_inputs = [torch_aie.Input(shape = input_shape, dtype = torch.int32, format = torch_aie.TensorFormat.ND), 
                      torch_aie.Input(shape = input_shape, dtype = torch.int32, format = torch_aie.TensorFormat.ND), 
                      torch_aie.Input(shape = input_shape, dtype = torch.int32, format = torch_aie.TensorFormat.ND)]
    compiled_model = torch_aie.compile(
        traced_model,
        inputs = compile_inputs,
        precision_policy = _enums.PrecisionPolicy.FP16,
        soc_version = "Ascend310P3",
        optimization_level = 0
    )
    print("torch_aie compile done !")
    compiled_model.save(ar.pt_dir)


    if ar.compare_cpu:
        print("start to check the percision of npu model.")
        com_res = True
        compiled_model = torch.jit.load(ar.pt_dir)
        jit_res = traced_model(input_ids, att_mask, token_ids)
        print("jit infer done !")
        aie_res = compiled_model(input_ids, att_mask, token_ids)
        print("aie infer done !")

        for j, a in zip(jit_res, aie_res):
            res = cosine_similarity(j, a)
            print(res)
            if res < COSINE_THRESHOLD:
                com_res = False

        if com_res:
            print("Compare success ! NPU model have the same output with CPU model !")
        else:
            print("Compare failed ! Outputs of NPU model are not the same with CPU model !")
    return 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32,
                        help="seq length for input data.")
    parser.add_argument("--prefix_dir", type=str, default='./albert_pytorch',
                        help="prefix dir for ori model code")
    parser.add_argument("--pth_dir", type=str, default='./albert_pytorch/outputs/SST-2',
                        help="dir of pth, load args.bin and model.bin")
    parser.add_argument("--data_dir", type=str, default='./albert_pytorch/dataset/SST-2',
                        help="dir of dataset")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="seq length for input data.")
    parser.add_argument("--save_dir", type=str, default='./',
                        help="save dir of model compiled by torch_aie")
    parser.add_argument("--compare_cpu", action='store_true',
                        help="Whether to check the percision of npu model.")
    ar = parser.parse_args()

    ar.pth_arg_path = os.path.join(ar.pth_dir, "training_args.bin")
    ar.data_type = 'dev'
    ar.model_name = "albert_seq{}_bs{}".format(ar.max_seq_length, ar.batch_size)
    ar.pt_dir = ar.save_dir + ar.model_name + ".pt"
    torch_model = get_torch_model(ar)
    aie_compile(torch_model, ar)


if __name__ == "__main__":
    main()