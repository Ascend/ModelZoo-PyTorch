from ais_bench.infer.interface import InferSession

import argparse
import time
from tqdm import tqdm
import torch
import numpy as np
import torch_aie
from torch_aie import _enums

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--ts_model', type=str)
    args.add_argument('--batch_size', default=1, type=int)
    args.add_argument('--optimization_level', default=0, type=int)
    return args.parse_args()

if __name__ == '__main__':
    infer_times = 100
    om_cost = 0
    pt_cost = 0
    opts = parse_args()
    
    ts_model = torch.jit.load(opts.ts_model)
    input_info = [torch_aie.Input((opts.batch_size,3,608,608))]
    torch_aie.set_device(0)
    torchaie_model = torch_aie.compile(
        ts_model,
        inputs=input_info,
        precision_policy=_enums.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        require_full_compilation=False,
        soc_version='Ascend310P3',
        optimization_level=opts.optimization_level
    )
    torchaie_model.eval()

    inference_time = []
    for _ in tqdm(range(0,infer_times)):
        dummy_input = np.random.randn(opts.batch_size,3,608,608).astype(np.float16)
        dummy_input = torch.Tensor(dummy_input)
        input_npu = dummy_input.to("npu:0")

        stream = torch_aie.npu.Stream("npu:0")
        with torch_aie.npu.stream(stream):
            start = time.time()
            output = torchaie_model(input_npu)
            stream.synchronize()
            cost = time.time() - start
            if _ >=5:
                inference_time.append(cost)

        #output = output.to("cpu")
    avg_inf_time = sum(inference_time)/len(inference_time)
    throughput = opts.batch_size / avg_inf_time
    print(f'the model of batchsize {opts.batch_size} throughput using pt-plugin is : {throughput}')
