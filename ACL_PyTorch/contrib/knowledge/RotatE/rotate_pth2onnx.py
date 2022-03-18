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
import pdb
import sys
sys.path.append(r'KnowledgeGraphEmbedding/codes/')
import numpy as np
import torch

import argparse

from model import KGEModel

def to_numpy32(tensor):
    return tensor.detach().cpu().numpy().astype(np.int32) if tensor.requires_grad else tensor.cpu().numpy().astype(np.int32)

def to_numpy64(tensor):
    return tensor.detach().cpu().numpy().astype(np.int64) if tensor.requires_grad else tensor.cpu().numpy().astype(np.int64)

def pth2onnx(input_file, output_file, bs, mode):
    kge_model = KGEModel(
        model_name='RotatE',
        nentity=14541,
        nrelation=237,
        hidden_dim=1000,
        gamma=9.0,
        double_entity_embedding=True,
        double_relation_embedding=False
    )

    checkpoint = torch.load(input_file, map_location='cpu')
    kge_model.load_state_dict(checkpoint['model_state_dict'])
    for param_tensor in kge_model.state_dict():
        print(param_tensor, "\t", kge_model.state_dict()[param_tensor].size())
    input_names = ["pos", "neg"]
    output_names = ["score"]
    dynamic_axes = {'pos': {0: '-1'}, 'neg': {0: '-1'}}
    # pdb.set_trace()
    head = torch.randint(0, 14541, (bs, 1))
    relation = torch.randint(0, 233, (bs, 1))
    tail = torch.randint(0, 14541, (bs, 1))
    input1 = []
    for j in range(bs):
        inp = []
        for i in range(14541):
            inp.append(i)
        input1.append(inp)
    negative_sample = torch.from_numpy(np.array(input1))

    positive_sample = torch.cat([head, relation, tail], dim=1)
    positive_sample = torch.from_numpy(to_numpy64(positive_sample))
    negative_sample = torch.from_numpy(to_numpy32(negative_sample))

    torch.onnx.export(kge_model, ((positive_sample, negative_sample), mode), output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names,  opset_version=11, verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='postprocess of r2plus1d')
    parser.add_argument('--pth_path', default=r'./checkpoint')
    parser.add_argument('--onnx_path', default=r'./kge_onnx_16_tail.onnx')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--mode', default=r'tail-batch', help='select head-batch or tail-batch')

    args = parser.parse_args()
    pth2onnx(args.pth_path, args.onnx_path, args.batch_size, args.mode)