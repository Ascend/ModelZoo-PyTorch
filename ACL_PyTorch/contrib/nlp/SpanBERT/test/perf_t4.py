# Copyright 2022 Huawei Technologies Co., Ltd
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
import onnxruntime as ort
import torch
import time
import cv2
import numpy as np
import argparse

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def make_inf_dummy_input(bs):
    org_input_ids = torch.ones(bs, 512).long()
    org_token_type_ids = torch.ones(bs, 512).long()
    org_input_mask = torch.ones(bs, 512).long()

    return (org_input_ids, org_token_type_ids, org_input_mask)


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=1, type=int, required=True)
    parser.add_argument("--path", default="./spanBert_dynamicbs.onnx", type=str, required=True)
    args = parser.parse_args()
    ort_session = ort.InferenceSession(args.path, providers=['CUDAExecutionProvider'])
    inf_dummy_input = make_inf_dummy_input(args.bs)
    inf_dummy_input = [t.cpu().numpy() for t in inf_dummy_input]
    totaltime = 0
    count = 10
    for i in range(count):
        t1 = time_sync()
        pred_onnx = ort_session.run(None, {'input_ids': inf_dummy_input[0], 'token_type_ids': inf_dummy_input[1],'attention_mask': inf_dummy_input[2]})
        t2 = time_sync()
        totaltime += (t2 - t1)
    onnx_meantime = totaltime / 10.0 * 1000
    fps = 1000 / (onnx_meantime / args.bs)
    print('spanBert inference by onnxruntime mean time is {}ms, bs{}, {}fps'.format(onnx_meantime, args.bs, fps))
