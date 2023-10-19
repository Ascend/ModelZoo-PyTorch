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
    return args.parse_args()

if __name__ == '__main__':
    infer_times = 100
    om_cost = 0
    pt_cost = 0
    opts = parse_args()

    ts_model = torch.jit.load(opts.ts_model)
    min_shape = (1, 3, 224, 224)
    max_shape = (32, 3, 224, 224)
    input_info = []
    input_info.append(torch_aie.Input(min_shape = min_shape, max_shape= max_shape))
    torch_aie.set_device(0)
    torchaie_model = torch_aie.compile(
        ts_model,
        inputs=input_info,
        precision_policy=_enums.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        soc_version='Ascend310P3',
        optimization_level=0
    )
    torchaie_model.eval()

    for _ in tqdm(range(0,infer_times)):
        dummy_input = np.random.randn(1,3,224,224).astype(np.float16)
        dummy_input = torch.Tensor(dummy_input)
        start = time.time()
        output = torchaie_model(dummy_input)
        cost = time.time() - start
        pt_cost += cost

    print(f'pt avg cost: {pt_cost/infer_times}')



