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
# OM_PATH = "/onnx/mobilenetv1/mobilenet-v1_bs1.om" # revise dynm
# TS_PATH = "/onnx/mobilenetv1/mobilenetv1.ts" # revise
INPUT_WIDTH = 224 # revise
INPUT_HEIGHT = 224 # revise

def parse_args():
    args = argparse.ArgumentParser(description="A program that operates in 'om' or 'ts' mode.")
    args.add_argument("--mode", choices=["om", "ts"], required=True, help="Specify the mode ('om' or 'ts').")
    args.add_argument('--om_path',help='MobilenetV1 om file path', type=str,
                        default='/onnx/mobilenetv1/mobilenet-v1_bs1.om'
                        )
    args.add_argument('--ts_path',help='MobilenetV1 ts file path', type=str,
                        default='/onnx/shufflenetv2/shufflenetv2.ts'
                        )
    args.add_argument("--batch_size", type=int, default=4, help="batch size.")
    args.add_argument("--opt_level", type=int, default=0, help="opt level.")
    return args.parse_args()

if __name__ == '__main__':
    infer_times = 100
    om_cost = 0
    pt_cost = 0
    opts = parse_args()
    OM_PATH = opts.om_path
    TS_PATH = opts.ts_path
    BATCH_SIZE = opts.batch_size
    OPTS_LEVEL = opts.opt_level
    warm_ctr = 10

    if opts.mode == "om":
        om_model = InferSession(0, OM_PATH)
        
        while warm_ctr:
            dummy_input = np.random.randn(BATCH_SIZE, 3, INPUT_WIDTH, INPUT_HEIGHT).astype(np.float32)
            output = om_model.infer([dummy_input], 'static', custom_sizes=90000000)
            warm_ctr -= 1
        
        for _ in tqdm(range(0, infer_times)):
            dummy_input = np.random.randn(BATCH_SIZE, 3, INPUT_WIDTH, INPUT_HEIGHT).astype(np.float32)
            start = time.time()
            output = om_model.infer([dummy_input], 'static', custom_sizes=90000000) # revise static
            # output = om_model.infer([dummy_input], 'dymshape', custom_sizes=4000) # revise dynm fp32为4个字节，输出为1x1000
            cost = time.time() - start
            om_cost += cost
        print(f"fps: {infer_times} * {BATCH_SIZE} / {om_cost : .3f} samples/s")
        print("om fps: ", infer_times * BATCH_SIZE / om_cost)

    if opts.mode == "ts":
        ts_model = torch.jit.load(TS_PATH)
        
        # revise static
        input_info = [torch_aie.Input((BATCH_SIZE, 3, INPUT_WIDTH, INPUT_HEIGHT))]
        
        # revise dynm
        # min_shape = (1, 3, INPUT_WIDTH, INPUT_HEIGHT)
        # max_shape = (32, 3, INPUT_WIDTH, INPUT_HEIGHT)
        # input_info = [torch_aie.Input(min_shape=(1,3,INPUT_WIDTH,INPUT_HEIGHT), max_shape=(32,3,INPUT_WIDTH,INPUT_HEIGHT))]
        
        # checkpoint = torch.load("/onnx/psenet/PSENet_for_PyTorch_1.2.pth", map_location='cpu')
        # checkpoint['state_dict'] = proc_nodes_module(checkpoint, 'state_dict')
        # # model = mobilenet.mobilenet_v2(pretrained = False)
        # model = resnet50()
        # model.load_state_dict(checkpoint['state_dict'])
        # model.eval()
        # print(model.forward(torch.ones(1, 3, 704, 1216)).shape)
        
        torch_aie.set_device(0)
        print("start compile")
        torchaie_model = torch_aie.compile(
            ts_model,
            inputs=input_info,
            precision_policy=_enums.PrecisionPolicy.FP16,
            # allow_tensor_replace_int=True,
            soc_version='Ascend310P3',
            optimization_level=OPTS_LEVEL,
        )
        print("end compile")
        torchaie_model.eval()
        
        dummy_input = np.random.randn(BATCH_SIZE, 3, INPUT_WIDTH, INPUT_HEIGHT).astype(np.float32)
        input_tensor = torch.Tensor(dummy_input)
        input_tensor = input_tensor.to("npu:0")
        loops = 100
        warm_ctr = 10
        
        default_stream = torch_aie.npu.default_stream()   
        time_cost = 0
        
        while warm_ctr:
            _ = torchaie_model(input_tensor)
            default_stream.synchronize()
            warm_ctr -= 1

        for i in range(loops):
            t0 = time.time()
            _ = torchaie_model(input_tensor)
            default_stream.synchronize()
            t1 = time.time()
            time_cost += (t1 - t0)
            # print(i)

        print(f"fps: {loops} * {BATCH_SIZE} / {time_cost : .3f} samples/s")
        print("torch_aie fps: ", loops * BATCH_SIZE / time_cost)
        from datetime import datetime
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print("Current Time:", formatted_time)


    # print(f'om avg cost: {om_cost/infer_times*1000} ms')
    torch_aie.finalize() # 必须加，否则AOE有概率coredump

