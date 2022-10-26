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
import os
import numpy as np
import tensorflow as tf
from typing import List

from .interface import BaseRunner
from .tf_graph_nodes import TfGraphNodes


class TFRunner(BaseRunner):
    def __init__(self, device_id, b_gpu=False):
        self.did = device_id if b_gpu else -1
        self.b_gpu = b_gpu
        self.config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=False)
        self.config.gpu_options.allow_growth = True
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.did)

    def infer(self, graph_nodes:TfGraphNodes, out_names:List[str],
              feed_dict=None) -> List[np.ndarray]:
        assert(isinstance(graph_nodes, TfGraphNodes))
        assert(feed_dict is None or isinstance(feed_dict, dict))
        with tf.Graph().as_default():
            tf.import_graph_def(graph_nodes._graph_def, name='')
            with tf.Session(config=self.config) as sess:
                feed_dict = self._get_feed_dict(graph_nodes) \
                    if feed_dict is None else feed_dict
                out_list = self._get_out_list(out_names)
                return sess.run(out_list, feed_dict=feed_dict)

    def _get_feed_dict(self, graph_nodes:TfGraphNodes):
        in_nodes = graph_nodes.get_net_input_nodes()
        feed_dict = {}
        for node in in_nodes:
            tensor_name = node.name + ":0"
            in_tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
            nd_array = node.get_rand_tensor(2.0)
            feed_dict[in_tensor] = nd_array
        return feed_dict

    def _get_out_list(self, out_names:List[str]):
        out_list = []
        for name in out_names:
            assert(isinstance(name, str))
            name = (name + ":0") if (":" not in name) else name
            out_tensor = tf.get_default_graph().get_tensor_by_name(name)
            out_list.append(out_tensor)
        return out_list
