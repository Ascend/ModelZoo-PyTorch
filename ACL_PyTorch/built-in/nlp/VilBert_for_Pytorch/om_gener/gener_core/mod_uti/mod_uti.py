# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#

# -*- coding:utf-8 -*-
import importlib


from .mod_param_uti import ModFMK
from ..mod_modify.interface import BaseGraph


def get_mod(m_paths: tuple, fmk: ModFMK) -> BaseGraph:
    """
    use get_third_party_mod_info to get m_paths and fmk
    """
    if (fmk == ModFMK.TF):
        graph = importlib.import_module("gener_core.mod_modify.tf_graph_nodes")
        return graph.TfGraphNodes(m_paths[0])
    elif (fmk == ModFMK.ONNX):
        ox_graph = importlib.import_module("gener_core.mod_modify.onnx_graph")
        return ox_graph.OXGraph(m_paths[0])
    else:
        raise RuntimeError("not supported fmk:{}".format(fmk))
