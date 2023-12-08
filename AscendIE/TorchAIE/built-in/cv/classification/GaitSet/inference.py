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

import argparse
import copy
import numpy as np
import os
import time
import torch
import torch_aie
from tqdm import tqdm

from GaitSet.model.network import SetNet


class wrapperNet(torch.nn.Module):
    def __init__(self, module):
        super(wrapperNet, self).__init__()
        self.module = module


def parse_args():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--input_path", default="CASIA-B-bin")
    parser.add_argument("--output_path", default="result")
    parser.add_argument("--save_output", type=bool, default=True)
    parser.add_argument("--input_shape", nargs="+", type=int, default=[1, 100, 64, 44])
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup_num", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument(
        "--ckpt_path",
        default="GaitSet/work/checkpoint/GaitSet/GaitSet_CASIA-B_73_False_256_0.2_128_full_30-80000-encoder.ptm",
    )
    args = parser.parse_args()
    return args


def compile_model(args):
    encoder = SetNet(args.hidden_dim).float()
    # 增加一层wrapper使得权重keys和模型能够匹配
    encoder = wrapperNet(encoder)
    ckpt = torch.load(args.ckpt_path, map_location=torch.device("cpu"))
    encoder.load_state_dict(ckpt)

    net_input = torch.randn(args.input_shape)
    ts_model = torch.jit.trace(encoder.module.eval(), net_input)

    compile_input = [torch_aie.Input(args.input_shape, dtype=torch.float)]
    torch_aie_model = torch_aie.compile(
        ts_model,
        inputs=compile_input,
        precision_policy=torch_aie.PrecisionPolicy.FP16,
    )
    return torch_aie_model


def get_input_info(args):
    batch = args.input_shape[0]
    file_path_list = []
    file_name_list = []
    for dir_path, _, file_names in os.walk(args.input_path):
        for file_name in file_names:
            file_path_list.append(os.path.join(dir_path, file_name))
            file_name_list.append(file_name)

    file_nums = len(file_path_list)
    iters = (file_nums + batch - 1) // batch
    pad = iters * batch - file_nums
    return file_path_list, file_name_list, iters, pad


def create_data_generator(args, file_path_list, file_name_list, iters, pad):
    batch = args.input_shape[0]
    file_nums = len(file_path_list)

    for i in range(iters):
        batch_data = []
        batch_names = []
        for j in range(batch):
            idx = batch * i + j
            if idx == file_nums:
                break
            batch_data.append(
                torch.from_numpy(np.fromfile(file_path_list[idx], np.float32)).view(
                    [1, *args.input_shape[1:]]
                )
            )
            batch_names.append(file_name_list[idx])
            idx += 1
        if (i == (iters - 1)) and (pad > 0):
            for _ in range(pad):
                batch_data.append(copy.deepcopy(batch_data[-1]))
                batch_names.append(file_name_list[-1])
        yield torch.cat(batch_data).to(args.device), batch_names


def inference(args, torch_aie_model):
    output_path = args.output_path
    batch = args.input_shape[0]
    if args.save_output and (not os.path.exists(output_path)):
        os.makedirs(output_path, mode=640)

    file_path_list, file_name_list, iters, pad = get_input_info(args)
    data_gernerator = create_data_generator(
        args, file_path_list, file_name_list, iters, pad
    )

    cumulated_time = 0
    counted_num = 0
    for x, x_names in tqdm(data_gernerator, total=iters):
        counted_num += 1
        stream = torch_aie.npu.Stream(args.device)
        with torch_aie.npu.stream(stream):
            stream.synchronize()
            ts = time.time()
            ret = torch_aie_model.forward(x)
            stream.synchronize()
            duration = time.time() - ts
            if counted_num > args.warmup_num:
                cumulated_time += duration

        feature = ret.detach().to("cpu").numpy()
        if args.save_output:
            for i in range(batch):
                feature[i, ...].tofile(
                    os.path.join(output_path, x_names[i].replace(".bin", "_0.bin"))
                )

    print(
        f"Pure inference performance: {batch * (counted_num - args.warmup_num) / cumulated_time} FPS"
    )


def main():
    args = parse_args()
    torch_aie.set_device(int(args.device.split(":")[-1]))
    torch_aie_model = compile_model(args)
    inference(args, torch_aie_model)


if __name__ == "__main__":
    main()
