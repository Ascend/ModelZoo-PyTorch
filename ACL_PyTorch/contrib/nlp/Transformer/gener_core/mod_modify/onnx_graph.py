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
from typing import List, Union, Set, Tuple
import onnx
from onnx import ModelProto
from onnx import NodeProto
from onnx import TensorProto
from onnx import ValueInfoProto
from onnx import helper
import numpy as np

from .interface import BaseGraph
from .interface import AttrType
from .onnx_node import OXNode
from .onnx_node import INPUT_TYPE
from .onnx_node import INIT_TYPE
from .onnx_node import DATA_TYPE_MAP


def _add_obj_to_dict(key, obj, res):
    if (key in res):
        raise RuntimeError("key {} exist. value:{}".format(key, obj))
    res[key] = obj


def _replace_input(node: OXNode, input_map: dict):
    # input_map is src_name, dst_name
    for idx in range(0, len(node.input_name)):
        if (node.input_name[idx] in input_map):
            node.node.input[idx] = input_map.get(node.input_name[idx])


def _gen_tensor_map(src_node: OXNode, dst_node: OXNode) -> dict:
    src_out = src_node.out_name
    dst_out = dst_node.out_name
    if (len(dst_out) != len(src_out)):
        raise RuntimeError("{} {} out num not match. {}, {}".format(
            src_node.name, dst_node.name, len(dst_out), len(src_out)))
    input_map = {}
    for src, dst in zip(src_out, dst_out):
        input_map[src] = dst
    return input_map


def _update_in_out_map(in_node: OXNode, rm_node: OXNode, in_out_map: dict):
    if (in_node.name not in in_out_map or rm_node.name not in in_out_map):
        raise RuntimeError("invalid in_out_map. {},{}".format(
            in_node.name, rm_node.name))
    in_node_out = in_out_map.get(in_node.name)
    if (rm_node.name not in in_node_out):
        raise RuntimeError("in/out not match {}, {}".format(
            in_node.name, rm_node.name))
    while rm_node.name in in_node_out:
        in_node_out.remove(rm_node.name)

    rm_out = in_out_map.get(rm_node.name)
    in_node_out.extend(rm_out)
    in_out_map[in_node.name] = in_node_out
    in_out_map.pop(rm_node.name)


def _clear_graph(tar_mod: ModelProto):
        while (len(tar_mod.graph.node) > 0):
            tar_mod.graph.node.pop()
        while (len(tar_mod.graph.input) > 0):
            tar_mod.graph.input.pop()
        while(len(tar_mod.graph.initializer) > 0):
            tar_mod.graph.initializer.pop()


class OXGraph(BaseGraph):
    def __init__(self, path: str):
        self.mod = onnx.load(path)
        self.name_node_map = {}
        self.remove_names = set()
        for node in self.mod.graph.node:
            cur_ox_node = OXNode(node)
            _add_obj_to_dict(node.name, cur_ox_node, self.name_node_map)
            for out_name in node.output:
                _add_obj_to_dict(out_name, cur_ox_node, self.name_node_map)
        for in_node in self.mod.graph.input:
            cur_ox_node = OXNode(in_node)
            _add_obj_to_dict(in_node.name, cur_ox_node, self.name_node_map)
        for node in self.mod.graph.initializer:
            cur_ox_node = OXNode(node)
            _add_obj_to_dict(node.name, cur_ox_node, self.name_node_map)

    def get_node(self, name: str) -> OXNode:
        if (isinstance(name, OXNode)):
            if (name.name in self.remove_names):
                raise RuntimeError("{} were removed".format(name.name))
            return name

        if (name in self.remove_names):
            raise RuntimeError("{} were removed".format(name))
        if (name not in self.name_node_map):
            raise ValueError("onnx graph not have node {}".format(name))
        return self.name_node_map.get(name)

    def get_nodes_by_optype(self, tar_optype) -> List[OXNode]:
        if (tar_optype == INIT_TYPE):
            tar_nodes = self.mod.graph.initializer
        elif (tar_optype == INPUT_TYPE):
            tar_nodes = self.mod.graph.input
        else:
            tar_nodes = self.mod.graph.node

        res = []
        for node in tar_nodes:
            node = self.get_node(node.name)
            if (node.op_type == tar_optype):
                res.append(node)
        return res

    def get_net_in_out_map(self) -> dict:
        # name as key, out names list as value
        res = {}
        for node in self.mod.graph.node:
            cur_out_name = node.name
            for in_name in node.input:
                cur_in_name = self.get_node(in_name).name
                if (cur_in_name not in res):
                    res[cur_in_name] = [cur_out_name]
                else:
                    res[cur_in_name].append(cur_out_name)
        return res

    def get_net_input_nodes(self) -> List[OXNode]:
        return self.get_nodes_by_optype(INPUT_TYPE)

    def get_net_output_nodes(self) -> List[OXNode]:
        res = []
        for node in self.mod.graph.output:
            cur_ox_node = self.get_node(node.name)
            res.append(cur_ox_node)
        return res

    def get_nodes_forward_node(self, node: Union[str, OXNode],
                               input_idx: Union[List[int], Tuple[int]] = None,
                               if_self: bool = True,
                               end_nodes: List[str] = None) -> Set[str]:
        start_node = self.get_node(node)
        end_names = set()
        if (end_nodes is not None):
            for end_n in end_nodes:
                end_n = self.get_node(end_n)
                end_names.add(end_n.name)
        start_in_num = len(start_node.input_name)
        if (input_idx is None):
            input_idx = [idx for idx in range(0, start_in_num)]

        res_nodes = set()
        if (if_self):
            res_nodes.add(start_node.name)
        candi_nodes = []
        for idx in input_idx:
            tmp_node = self.get_node(start_node.input_name[idx])
            candi_nodes.append(tmp_node)
        while (len(candi_nodes) > 0):
            tmp_node = candi_nodes.pop()
            if (tmp_node.name in res_nodes):
                continue
            res_nodes.add(tmp_node.name)
            if (tmp_node.name not in end_names):
                new_in = [self.get_node(name) for name in tmp_node.input_name]
                candi_nodes.extend(new_in)
        return res_nodes

    def get_nodes_behind_node(self, node: Union[str, OXNode],
                              if_self: bool = False,
                              end_nodes: List[str] = None,
                              in_out_map: dict = None) -> Set[str]:
        end_names = set()
        if (end_nodes is not None):
            for end_n in end_nodes:
                end_n = self.get_node(end_n)
                end_names.add(end_n.name)
        start_node = self.get_node(node)

        if (in_out_map is None):
            in_out_map = self.get_net_in_out_map()

        res_nodes = set()
        if (if_self):
            res_nodes.add(start_node.name)
        candi_names = in_out_map.get(start_node.name)
        if (candi_names is None):
            return res_nodes

        while (len(candi_names) > 0):
            tmp_node = self.get_node(candi_names.pop())
            if (tmp_node.name in res_nodes):
                continue
            res_nodes.add(tmp_node.name)
            if (tmp_node.name not in end_names):
                new_out = in_out_map.get(tmp_node.name)
                if (new_out is not None):
                    candi_names.extend(new_out)
        return res_nodes

    def add_placeholder_node(self, name: str, data_type: str,
                             shape: Union[List, Tuple]) -> OXNode:
        if (name in self.name_node_map):
            raise RuntimeError("new node name:{} exist".format(name))
        new_in = self.mod.graph.input.add()
        new_ox = OXNode(new_in)
        new_ox.set_name(name)
        new_ox.set_attr({"dtype": (AttrType.DTYPE, data_type),
                         "shape": (AttrType.SHAPE, shape)})
        self._update_name_node_map(new_ox)
        return new_ox

    def add_const_node(self, name: str, value: np.ndarray) -> OXNode:
        if (not isinstance(value, np.ndarray)):
            raise TypeError("const value not ndarray")
        return self.add_new_node(name, "Constant",
                                 {"value": (AttrType.TENSOR, value)})

    def add_new_node(self, name, op_type, attrs: dict = None, out_num = 1):
        if (name in self.name_node_map):
            raise RuntimeError("new node name:{} exist".format(name))
        new_node = self.mod.graph.node.add()
        new_ox = OXNode(new_node)
        new_ox.set_name(name)
        new_ox.set_op(op_type)
        new_ox.set_attr(attrs if attrs is not None else {})
        new_ox.set_out_name(out_num)
        self._update_name_node_map(new_ox)
        return new_ox

    def node_remove(self, nodes_name: Union[List, Tuple, Set]):
        # usually for node has no input
        for node in nodes_name:
            ox_node = self.get_node(node)
            if (ox_node.op_type == INPUT_TYPE):
                self.mod.graph.input.remove(ox_node.node)
                self.remove_names.add(ox_node.name)
            elif (ox_node.op_type == INIT_TYPE):
                self.mod.graph.initializer.remove(ox_node.node)
                self.remove_names.add(ox_node.name)
            else:
                self.mod.graph.node.remove(ox_node.node)
                self.remove_names.add(ox_node.name)
                for rm_name in ox_node.out_name:
                    self.remove_names.add(rm_name)

    def node_remove_connect(self, node_remove: Union[str, OXNode], in_idx=0,
                            in_out_map: dict = None):
        rm_node = self.get_node(node_remove)
        if (in_idx >= len(rm_node.input_name)):
            raise ValueError("in idx {} exceeds input num:{}".format(
                in_idx, len(rm_node.input_name)))
        in_node = self.get_node(rm_node.input_name[in_idx])
        input_map = _gen_tensor_map(rm_node, in_node)

        tar_nodes = self.mod.graph.node
        if (isinstance(in_out_map, dict)):
            tar_nodes = in_out_map.get(rm_node.name)

        for node in tar_nodes:
            if (isinstance(node, NodeProto)):
                node = node.name
            node = self.get_node(node)
            _replace_input(node, input_map)
        self.node_remove((rm_node,))

        if (isinstance(in_out_map, dict)):
            _update_in_out_map(in_node, rm_node, in_out_map)

    def node_replace(self, node_original: Union[str, OXNode],
                     node_new: Union[str, OXNode],
                     input_node: List[Union[str, OXNode]] = None,
                     in_node_output_index: List[int] = None,
                     if_remove: bool = True,
                     in_out_map: dict = None):
        node_ori = self.get_node(node_original)
        node_new = self.get_node(node_new)
        if (input_node is None):
            node_new.copy_input_from_node(node_ori)
        else:
            node_new.set_input_node(0, input_node, in_node_output_index)
        input_map = _gen_tensor_map(node_ori, node_new)

        tar_nodes = self.mod.graph.node
        if (isinstance(in_out_map, dict)):
            tar_nodes = in_out_map.get(node_ori.name)

        for node in tar_nodes:
            if (isinstance(node, NodeProto)):
                node = node.name
            node = self.get_node(node)
            _replace_input(node, input_map)
        if (if_remove):
            self.node_remove((node_ori,))

        if (isinstance(in_out_map, dict)):
            self._update_io_map_by_replace(node_ori, node_new,
                                           in_out_map, if_remove)

    def node_add_forward(self, node_original: Union[str, OXNode],
                         node_add: Union[str, OXNode]):
        node_ori = self.get_node(node_original)
        node_add = self.get_node(node_add)
        ori_in_num = len(node_ori.input_name)
        if (ori_in_num <= 0):
            raise RuntimeError("node {} has no input".format(node_ori.name))
        if (len(node_add.out_name) != ori_in_num):
            raise RuntimeError("out num {} not match in num {}. {}, {}".format(
                len(node_add.out_name), len(node_ori.input_name),
                node_add.name, node_ori.name))
        node_add.copy_input_from_node(node_ori)
        node_ori.set_input_node(0, [node_add for idx in range(0, ori_in_num)],
                                [idx for idx in range(0, ori_in_num)])

    def node_add_behind(self, node_original: Union[str, OXNode],
                        node_add: Union[str, OXNode],
                        in_out_map: dict = None):
        node_ori = self.get_node(node_original)
        node_add = self.get_node(node_add)
        ori_out_num = len(node_ori.out_name)
        if (ori_out_num <= 0):
            raise RuntimeError("node {} has no output".format(node_ori.name))
        node_add.set_input_node(0, [node_ori for idx in range(0, ori_out_num)],
                                [idx for idx in range(0, ori_out_num)])
        input_map = _gen_tensor_map(node_ori, node_add)

        tar_nodes = self.mod.graph.node
        if (isinstance(in_out_map, dict)):
            tar_nodes = in_out_map.get(node_ori.name)

        for node in tar_nodes:
            if (isinstance(node, NodeProto)):
                node = node.name
            node = self.get_node(node)
            _replace_input(node, input_map)

        if (isinstance(in_out_map, dict)):
            self._update_io_map_by_add_behind(node_ori, node_add, in_out_map)

    def save_new_model(self, new_mod_path: str,
                       nodes_save: Union[Set, List] = None):
        if (nodes_save is None):
            onnx.save(self.mod, new_mod_path)
            return

        tar_names = set()
        for node in nodes_save:
            node = self.get_node(node)
            tar_names.add(node.name)

        tar_mod = ModelProto()
        tar_mod.CopyFrom(self.mod)
        _clear_graph(tar_mod)
        for name, node in self.name_node_map.items():
            if (name not in tar_names):
                continue
            if (node.op_type == INPUT_TYPE):
                new_node = ValueInfoProto()
                new_node.CopyFrom(node.node)
                tar_mod.graph.input.append(new_node)
            elif (node.op_type == INIT_TYPE):
                new_node = TensorProto()
                new_node.CopyFrom(node.node)
                tar_mod.graph.initializer.append(new_node)
            else:
                new_node = NodeProto()
                new_node.CopyFrom(node.node)
                tar_mod.graph.node.append(new_node)
        onnx.save(tar_mod, new_mod_path)

    def add_output_node(self, name, dtype="float32"):
        if (name not in self.name_node_map):
            raise ValueError("name:{} not in mod".format(name))
        dtype = DATA_TYPE_MAP.get(dtype)
        cur_tensor = helper.make_tensor_value_info(name, dtype, None)
        self.mod.graph.output.append(cur_tensor)

    def clear_output(self):
        while (len(self.mod.graph.output) > 0):
            self.mod.graph.output.pop()

    def _update_name_node_map(self, node: OXNode):
        _add_obj_to_dict(node.name, node, self.name_node_map)
        if (node.op_type in (INPUT_TYPE, INIT_TYPE)):
            return
        for name in node.out_name:
            _add_obj_to_dict(name, node, self.name_node_map)

    def _update_io_map_by_input(self, node: OXNode, io_map: dict):
        for in_name in node.input_name:
            in_node = self.get_node(in_name)
            if (in_node.name not in io_map):
                io_map[in_node.name] = [node.name]
            else:
                if (node.name not in io_map[in_node.name]):
                    io_map[in_node.name].append(node.name)

    def _update_io_map_by_output(self, node: OXNode, io_map: dict, src_out):
        if (node.name in io_map):
            dst_out = io_map.get(node.name)
            dst_out.extend(src_out)
            io_map[node.name] = dst_out
        else:
            io_map[node.name] = src_out

    def _update_io_map_by_replace(self, src_node: OXNode, dst_node: OXNode,
                                  io_map: dict, if_remove=True):
        if (src_node.name not in io_map):
            raise RuntimeError("{} not in in_out_map".format(src_node.name))

        self._update_io_map_by_input(dst_node, io_map)
        src_out = io_map.get(src_node.name).copy()
        self._update_io_map_by_output(dst_node, io_map, src_out)

        if (if_remove):
            io_map.pop(src_node.name)

    def _update_io_map_by_add_behind(self, node_ori: OXNode, node_add: OXNode,
                                     io_map: dict):
        if (node_ori.name not in io_map):
            raise RuntimeError("{} not in in_out_map".format(node_ori.name))
        self._update_io_map_by_input(node_add, io_map)
        ori_out = io_map.get(node_ori.name).copy()
        self._update_io_map_by_output(node_add, io_map, ori_out)
        io_map[node_ori.name] = [node_add.name]
