# -*- coding:utf-8 -*-
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
from typing import List
import os
import numpy as np
import onnx
from onnx import ModelProto
import onnxruntime as onrt

from .interface import BaseRunner
from .onnx_graph import OXGraph


class OXRunner(BaseRunner):
    def __init__(self, device_id, b_gpu=False):
        self.did = device_id if b_gpu else -1
        self.b_gpu = b_gpu

    def infer(self, graph: OXGraph, out_names: List[str],
              dtypes: List[str]=None, feed_dict=None) -> List[np.ndarray]:
        tmp_name = "_gc_tmp_run.onnx"
        ori_mod = ModelProto()
        ori_mod.CopyFrom(graph.mod)
        if (not isinstance(graph, OXGraph)):
            raise TypeError("input graph not OXGraph")
        if (feed_dict is not None and not isinstance(feed_dict, dict)):
            raise TypeError("invalid feed_dict. {}".format(type(feed_dict)))
        fd = self._get_feed_dict(graph) if feed_dict is None else feed_dict

        out_names, out_nums, dtypes = self._gen_out_info(graph,
                                                         out_names, dtypes)
        graph.clear_output()
        for name, dtype in zip(out_names, dtypes):
            graph.add_output_node(name, dtype)
        onnx.save(graph.mod, tmp_name)
        graph.mod.CopyFrom(ori_mod)
        sess = onrt.InferenceSession(tmp_name)
        out_tensors = sess.run(out_names, fd)
        os.remove(tmp_name)
        res = []
        b_idx = 0
        for num in out_nums:
            res.append(out_tensors[b_idx: b_idx+num])
            b_idx += num
        return res

    def _get_feed_dict(self, graph: OXGraph):
        in_nodes = graph.get_net_input_nodes()
        feed_dict = {}
        for node in in_nodes:
            nd_array = node.get_rand_tensor(2.0)
            feed_dict[node.name] = nd_array
        return feed_dict

    def _gen_out_info(self, graph, out_names, dtypes):
        out_tensor_names = []
        out_nums = []
        for name in out_names:
            node = graph.get_node(name)
            cur_out = node.out_name
            out_tensor_names.extend(cur_out)
            out_nums.append(len(cur_out))

        if (dtypes is not None):
            if (len(dtypes) != len(out_tensor_names)):
                raise RuntimeError("length not match:{}, {}".format(
                    len(dtypes), len(out_tensor_names)))
        else:
            dtypes = ["float32" for name in out_tensor_names]
        return out_tensor_names, out_nums, dtypes
