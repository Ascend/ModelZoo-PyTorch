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

import argparse

from auto_optimizer import OnnxGraph


def insert_grid_sample(in_onnx, out_onnx):
    model = OnnxGraph.parse(in_onnx)
    model = model.simplify(skip_shape_inference=True)
    
    # 1. delet "Unsqueeze_Pad_Squeeze_Transpose_Add"
    pad = model.get_nodes('Pad')[0]
    unsq = model.get_prev_node(pad.inputs[0])
    sq = model.get_next_nodes(pad.outputs[0])[0]
    trans = model.get_next_nodes(sq.outputs[0])[0]
    ad = model.get_next_nodes(trans.outputs[0])[0]
    inputs = [None, unsq.inputs[0]]
    for inp in ad.inputs:
        if inp != trans.outputs[0]:
            inputs[0] = inp
    outputs = ad.outputs
    model.remove(unsq.name, {})
    model.remove(pad.name, {})
    model.remove(sq.name, {})
    model.remove(trans.name, {})
    model.remove(ad.name, {})

    # 2. insert grid_sample\
    model.add_node('GridSample_0', 'GridSample',
                    attrs={'align_corners': 1, 'mode': 'bilinear', 'padding_mode': 'zeros'},
                    inputs=inputs, outputs=outputs)

    model.update_map()
    model.save(out_onnx)
    
    print("[info] Insert grid_sample done.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="opt onnx")  # task process paramater
    parser.add_argument('--in_onnx', type=str)
    parser.add_argument('--out_onnx', type=str)
    args = parser.parse_args()
    
    insert_grid_sample(args.in_onnx, args.out_onnx)
    print("[info] Optimize onnx success. result onnx is: {}".format(args.out_onnx))
    