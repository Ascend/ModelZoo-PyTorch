import argparse
import time

import torch
import numpy as np
from tqdm import tqdm

from ais_bench.infer.interface import InferSession
import torch_aie
from torch_aie import _enums

INPUT_WIDTH = 128
INPUT_HEIGHT = 128

def parse_args():
    args = argparse.ArgumentParser(description="A program that operates in 'om' or 'ts' mode.")
    args.add_argument("--mode", choices=["om", "ts"], required=True, help="Specify the mode ('om' or 'ts').")
    args.add_argument('--om_path',help='MobilenetV1 om file path', type=str,
                        default='/onnx/mobilenetv1/mobilenet-v1_bs1.om'
                        )
    args.add_argument('--ts_path',help='MobilenetV1 ts file path', type=str,
                        default='/onnx/stargan/stargan.ts'
                        )
    args.add_argument("--batch-size", type=int, default=4, help="batch size.")
    return args.parse_args()


if __name__ == '__main__':
    infer_times = 100
    om_cost = 0
    pt_cost = 0
    opts = parse_args()
    OM_PATH = opts.om_path
    TS_PATH = opts.ts_path
    BATCH_SIZE = opts.batch_size

    if opts.mode == "om":
        om_model = InferSession(0, OM_PATH)
        for _ in tqdm(range(0, infer_times)):
            dummy_input = np.random.randn(BATCH_SIZE, 3, INPUT_WIDTH, INPUT_HEIGHT).astype(np.uint8)
            start = time.time()
            output = om_model.infer([dummy_input], 'static', custom_sizes=90000000)
            cost = time.time() - start
            om_cost += cost

    if opts.mode == "ts":
        ts_model = torch.jit.load(TS_PATH)
        
        input_info = [torch_aie.Input((BATCH_SIZE, 3, INPUT_WIDTH, INPUT_HEIGHT)), torch_aie.Input((BATCH_SIZE, 5))]
        
        torch_aie.set_device(0)
        print("start compile")
        torchaie_model = torch_aie.compile(
            ts_model,
            inputs=input_info,
            precision_policy=_enums.PrecisionPolicy.FP16,
            soc_version='Ascend310P3',
        )
        print("end compile")
        torchaie_model.eval()
        
        dummy_input = np.zeros((BATCH_SIZE, 3, INPUT_WIDTH, INPUT_HEIGHT)).astype(np.float32)
        input_tensor = torch.Tensor(dummy_input)
        input_tensor = input_tensor.to("npu:0")
        dummy_input2 = np.ones((BATCH_SIZE, 5)).astype(np.int32)
        input_tensor2 = torch.Tensor(dummy_input2)
        input_tensor2 = input_tensor2.to("npu:0")
        loops = 100
        warm_ctr = 10
        
        default_stream = torch_aie.npu.default_stream()   
        time_cost = 0
        
        while warm_ctr:
            _ = torchaie_model(input_tensor, input_tensor2)
            default_stream.synchronize()
            warm_ctr -= 1

        for i in range(loops):
            t0 = time.time()
            _ = torchaie_model(input_tensor, input_tensor2)
            default_stream.synchronize()
            t1 = time.time()
            time_cost += (t1 - t0)
            print(i)

        print(f"fps: {loops} * {BATCH_SIZE} / {time_cost : .3f} samples/s")
        print("torch_aie fps: ", loops * BATCH_SIZE / time_cost)

    from datetime import datetime
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Current Time:", formatted_time)
