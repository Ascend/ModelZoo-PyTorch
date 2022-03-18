# -*- coding: utf-8 -*-

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import sklearn
import torch
import torch.onnx

from collections import OrderedDict
from deepctr_torch.models import WDL
from deepctr_torch.inputs import SparseFeat, DenseFeat


def proc_nodes_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(pth_file_path, onnx_file_path):
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    checkpoint['state_dict'] = proc_nodes_module(checkpoint, 'state_dict')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    sparse_nunique = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992,
                      5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]
    fixlen_feature_columns = [SparseFeat(feat, sparse_nunique[idx], embedding_dim=4)
                              for idx, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    model = WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                task='binary', dnn_hidden_units=(512, 256, 128))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]

    sparse_dummy_input = []
    for nunique in sparse_nunique:
        sparse_dummy_input.append(torch.randint(0, nunique, size=[1, 1]))
    sparse_dummy_input = torch.cat(sparse_dummy_input, dim=-1)
    dummy_input = torch.cat([sparse_dummy_input.float(), torch.randn(size=[1, len(dense_features)])], dim=-1)

    torch.onnx.export(model, dummy_input, onnx_file_path, input_names=input_names, output_names=output_names,
                      opset_version=11)

if __name__ == '__main__':
    src_file_path = "checkpoint.pth.tar"
    dst_file_path = "checkpoint.onnx"

    convert(src_file_path, dst_file_path)
