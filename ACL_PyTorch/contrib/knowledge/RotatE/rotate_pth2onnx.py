# Copyright 2022 Huawei Technologies Co., Ltd
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


import torch
import argparse
import sys
sys.path.append(r'KnowledgeGraphEmbedding/codes/')
from model import KGEModel


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

    head = torch.randint(0, 14541, (bs, 1))
    relation = torch.randint(0, 233, (bs, 1))
    tail = torch.randint(0, 14541, (bs, 1))
    positive_sample = torch.cat([head, relation, tail], dim=1)
    negative_sample = torch.arange(14541).tile(bs, 1).int()

    torch.onnx.export(kge_model, ((positive_sample, negative_sample), mode), output_file,
                      input_names=["pos", "neg"],
                      dynamic_axes={'pos': {0: '-1'}, 'neg': {0: '-1'}},
                      output_names=["score"],
                      opset_version=11)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RotatE')
    parser.add_argument('--pth_path', default=r'./checkpoint')
    parser.add_argument('--onnx_path', default=r'./kge_onnx_16_tail.onnx')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--mode', default=r'tail-batch',
                        help='select head-batch or tail-batch')

    args = parser.parse_args()
    pth2onnx(args.pth_path, args.onnx_path, args.batch_size, args.mode)
