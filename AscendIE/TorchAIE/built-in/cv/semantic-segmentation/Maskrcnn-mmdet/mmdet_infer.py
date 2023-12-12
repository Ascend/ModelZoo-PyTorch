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
import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch_aie


class cocoDataset(Dataset):
    def __init__(self, args):
        super(cocoDataset, self).__init__()

        self.dataDir = args.infer_data
        self.inputShape = args.input_shape
        self.dataList = sorted([file for file in os.listdir(args.infer_data)])

    def __getitem__(self, index):
        dataName = self.dataList[index]
        data = np.fromfile(os.path.join(self.dataDir, dataName), dtype=np.float32)
        data = np.reshape(data, self.inputShape)
        return data, dataName
    
    def __len__(self):
        return len(self.dataList)


def inferModel(args, aieModel):
    print("Begin model inference")
    
    cocoData = cocoDataset(args)
    cocoLoader = DataLoader(cocoData, batch_size=args.batch_size)

    infer_time = []
    for index, (data, dataName) in tqdm(enumerate(cocoLoader)):
        data = data.to("npu:" + str(args.device))

        stream = torch_aie.npu.Stream("npu:" + str(args.device))
        with torch_aie.npu.stream(stream):
            inf_start = time.time()
            output = aieModel.forward(data)
            stream.synchronize()
            inf_end = time.time()
            inf = inf_end - inf_start
            if index >= args.warmup_num:
                infer_time.append(inf)

        if args.precision:
            savePath = os.path.join(args.infer_result, dataName[0][:-4])
            output_0 = output[0][0].to("cpu").numpy().astype(np.float32)
            output_1 = output[1][0].to("cpu").numpy().astype(np.int32)
            output_2 = output[2][0].to("cpu").numpy().astype(np.float32)
            output_0.tofile(savePath + "_0.bin")
            output_1.tofile(savePath + "_1.bin")
            output_2.tofile(savePath + "_2.bin")

    avg_inf_time = sum(infer_time) / len(infer_time)
    throughput = args.batch_size / avg_inf_time

    print("Average throughput with batch_size={} is {:.3f} FPS".format(args.batch_size, throughput))


def inferModelWithZeros(args, aieModel):
    print("Begin model inference with zeros data")

    infer_num = 100
    infer_time = []
    inputShape = [args.batch_size] + args.input_shape
    for index in tqdm(range(infer_num)):
        data = torch.zeros(inputShape, dtype=torch.float32).to("npu:" + str(args.device))

        stream = torch_aie.npu.Stream("npu:" + str(args.device))
        with torch_aie.npu.stream(stream):
            inf_start = time.time()
            output = aieModel.forward(data)
            stream.synchronize()
            inf_end = time.time()
            inf = inf_end - inf_start
            if index >= args.warmup_num:
                infer_time.append(inf)

    avg_inf_time = sum(infer_time) / len(infer_time)
    throughput = args.batch_size / avg_inf_time

    print("Average throughput with batch_size={} is {:.3f} FPS".format(args.batch_size, throughput))


def compileModel(args):
    print("Begin compiling model with batch_size {}".format(args.batch_size))
    torchModel = torch.jit.load(args.torch_model)
    torchModel.eval()

    inputShape = [args.batch_size] + args.input_shape
    input = [torch_aie.Input(inputShape)]

    aieModel = torch_aie.compile(
        torchModel,
        inputs=input,
        precision_policy=torch_aie.PrecisionPolicy.FP16,
        optimization_level=0)
    print("Finish compiling")

    return aieModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_model", type=str, default="./mmdet_torch.ts")
    parser.add_argument("--aie_model", type=str, default="./mmdet_aie.ts")
    parser.add_argument("--infer_data", type=str, default="./val2017_bin")
    parser.add_argument("--infer_result", type=str, default="./result")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_shape", nargs="+", type=int, default=[3, 1216, 1216])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup_num", type=int, default=5)
    parser.add_argument("--precision", action="store_true")
    parser.add_argument("--reload_aie_model", action="store_true")
    parser.add_argument("--infer_with_zeros", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.infer_result):
        os.makedirs(args.infer_result)
    torch_aie.set_device(args.device)

    if args.reload_aie_model:
        aieModel = torch.jit.load(args.aie_model)
        print("Reload aie model from {}".format(args.aie_model))
    else:
        aieModel = compileModel(args)
        aieModel.save(args.aie_model)
        print("Saving aie model in {}".format(args.aie_model))

    if args.infer_with_zeros:
        inferModelWithZeros(args, aieModel)
    else:
        inferModel(args, aieModel)


if __name__ == "__main__":
    main()
