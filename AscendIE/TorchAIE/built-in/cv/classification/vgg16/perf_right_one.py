"""
切python2.0.1环境
"""

import argparse
import time
from tqdm import tqdm

import torch
import numpy as np

import torch_aie
from torch_aie import _enums

from ais_bench.infer.interface import InferSession

# OM_PATH = "/home/devkit1/cgznb1/vgg16-torchaie/vgg16_bs1.om" # revise static
# OM_PATH = "/onnx/vgg16-torchaie/vgg16_bs1.om" # revise dynm
# TS_PATH = "/onnx/vgg16-torchaie/vgg16.ts" # revise
INPUT_WIDTH = 224 # revise
INPUT_HEIGHT = 224 # revise

def parse_args():
    args = argparse.ArgumentParser(description="A program that operates in 'om' or 'ts' mode.")
    args.add_argument("--mode", choices=["om", "ts"], required=True, help="Specify the mode ('om' or 'ts').")
    args.add_argument('--om_path',help='MobilenetV1 om file path', type=str,
                        default='/onnx/mobilenetv1/mobilenet-v1_bs1.om'
                        )
    args.add_argument('--ts_path',help='MobilenetV1 ts file path', type=str,
                        default='/onnx/vgg16-torchaie/vgg16.ts'
                        )
    return args.parse_args()

if __name__ == '__main__':
    infer_times = 100
    om_cost = 0
    pt_cost = 0
    opts = parse_args()
    OM_PATH = opts.om_path
    TS_PATH = opts.ts_path

    if opts.mode == "om":
        om_model = InferSession(0, OM_PATH)
        for _ in tqdm(range(0, infer_times)):
            dummy_input = np.random.randn(1, 3, INPUT_WIDTH, INPUT_HEIGHT).astype(np.float32)
            start = time.time()
            output = om_model.infer([dummy_input], 'static', custom_sizes=90000000) # revise static
            # output = om_model.infer([dummy_input], 'dymshape', custom_sizes=4000) # revise dynm fp32为4个字节，输出为1x1000
            cost = time.time() - start
            om_cost += cost

    if opts.mode == "ts":
        ts_model = torch.jit.load(TS_PATH)
        
        # revise static
        input_info = [torch_aie.Input((1, 3, INPUT_WIDTH, INPUT_HEIGHT))]
        
        torch_aie.set_device(0)
        print("start compile")
        torchaie_model = torch_aie.compile(
            ts_model,
            inputs=input_info,
            precision_policy=_enums.PrecisionPolicy.FP32,
            # allow_tensor_replace_int=True,
            soc_version='Ascend310P3',
            # optimization_level=2
        )
        print("end compile")
        torchaie_model.eval()
        
        dummy_input = np.random.randn(1, 3, INPUT_WIDTH, INPUT_HEIGHT).astype(np.float32)
        input_tensor = torch.Tensor(dummy_input)
        loops = 100
        warm_ctr = 10
        batch_size = 1
        
        default_stream = torch_aie.npu.default_stream()   
        time_cost = 0
        
        input_tensor = input_tensor.to("npu:0")
        while warm_ctr:
            _ = torchaie_model(input_tensor)
            default_stream.synchronize()
            warm_ctr -= 1

        print("send to npu")
        input_tensor = input_tensor.to("npu:0")
        print("finish sent")
        for i in range(loops):
            t0 = time.time()
            _ = torchaie_model(input_tensor)
            default_stream.synchronize()
            t1 = time.time()
            time_cost += (t1 - t0)

        print(f"fps: {loops} * {batch_size} / {time_cost : .3f} samples/s")
        print("torch_aie fps: ", loops * batch_size / time_cost)

