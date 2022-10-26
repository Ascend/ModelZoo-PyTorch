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
from typing import List, Union, Tuple, Set
from enum import Enum
import numpy as np
from abc import ABCMeta


class AttrType(Enum):
    BOOL = 0
    FLOAT = 1
    INT = 2
    BYTE = 3                # func bytes. data_format = bytes("NCHW", "utf-8")
    SHAPE = 4               # list, tuple, ndarray
    TENSOR = 5              # list, tuple, ndarray
    DTYPE = 6               # np.xx or str xx. e.g. np.int32 or "int32"
    LIST_BOOL = 100         # add list value after LIST_BOOL
    LIST_FLOAT = 101
    LIST_INT = 102
    LIST_BYTE = 103
    LIST_SHAPE = 104
    LIST_TENSOR = 105
    LIST_DTYPE = 106

    def is_list(self):
        return self.value >= AttrType.LIST_BOOL.value

    def base_type(self):
        return AttrType(self.value - AttrType.LIST_BOOL.value) \
            if self.is_list() else self

    def list_type(self):
        return self if self.is_list() else \
            AttrType(self.value + AttrType.LIST_BOOL.value)


class BaseNode(object):
    __metaclass__ = ABCMeta

    @property
    def op_type(self) -> str:
        pass

    def set_op(self, op_type: str):
        pass

    @property
    def name(self) -> str:
        pass

    def set_name(self, name: str):
        pass

    def get_attr(self, name: str, attr_type: AttrType):
        pass

    def get_multi_attr(self, name_types: dict) -> dict:
        # input:  name as key, AttrType as value
        # output: name as key, value as value
        pass

    def set_attr(self, attr_infos: dict):
        # name as key, (AttrType, value) as value
        pass

    @property
    def shape(self) -> List[int]:           # for placeholder or input node
        pass

    def set_shape(self, shape: List[int]):
        pass

    def get_rand_tensor(self, rand_range) -> np.ndarray:
        pass

    @property
    def const_value(self) -> np.ndarray:    # for const node
        pass

    def set_const_value(self, value: np.array):
        pass

    @property
    def input_name(self) -> List[str]:
        pass

    def set_input_node(self, start_index: int, node_all: Union[List, Tuple],
                       node_all_output_index: List[int] = None):
        pass


class BaseGraph(object):
    __metaclass__ = ABCMeta

    def get_node(self, name: str) -> BaseNode:
        pass

    def get_nodes_by_optype(self, tar_optype) -> List[BaseNode]:
        pass

    def get_net_in_out_map(self) -> dict:
        # name as key, out names list as value
        pass

    def get_net_input_nodes(self) -> List[BaseNode]:
        pass

    def get_net_output_nodes(self) -> List[BaseNode]:
        pass

    def get_nodes_forward_node(self, node: Union[str, BaseNode],
                               input_idx: Union[List[int], Tuple[int]] = None,
                               if_self: bool = True,
                               end_nodes: List[str] = None) -> Set[str]:
        pass

    def get_nodes_behind_node(self, node: Union[str, BaseNode],
                              if_self: bool = False,
                              end_nodes: List[str] = None,
                              in_out_map: dict = None) -> Set[str]:
        pass

    def add_placeholder_node(self, name: str, data_type: str,
                             shape: Union[List, Tuple]) -> BaseNode:
        pass

    def add_const_node(self, name: str, value: np.ndarray) -> BaseNode:
        pass

    def add_new_node(self, name, op_type: str, attrs: dict = None,
                     out_num=1) -> BaseNode:
        pass

    def node_remove(self, nodes_name: Union[List, Tuple, Set]):
        # usually for node has no input
        pass

    def node_remove_connect(self, node_remove: Union[str, BaseNode],
                            in_out_map: dict = None):
        pass

    def node_remove_from(self, start_node: Union[str, BaseNode], if_self=True):
        in_out_map = self.get_net_in_out_map()
        start_node = self.get_node(start_node)
        start_name = start_node.name
        candi_names = start_node.input_name
        names_to_remove = set([start_name])
        for tar_name in candi_names:
            if (tar_name in in_out_map):
                cur_out = set(in_out_map[tar_name])
            else:
                tar_name = self.get_node(tar_name).name
                cur_out = set(in_out_map[tar_name])
            if (cur_out.issubset(names_to_remove)):
                names_to_remove.add(tar_name)
                tar_node = self.get_node(tar_name)
                candi_names.extend(tar_node.input_name)
        if not if_self:
            names_to_remove = names_to_remove-{start_name}
        self.node_remove(names_to_remove)
        return names_to_remove

    def node_replace(self, node_original: Union[str, BaseNode],
                     node_new: Union[str, BaseNode],
                     input_node: List[Union[str, BaseNode]] = None,
                     if_remove: bool = True,
                     in_out_map: dict = None):
        pass

    def node_add_forward(self, node_original: Union[str, BaseNode],
                         node_add: Union[str, BaseNode]):
        pass

    def node_add_behind(self, node_original: Union[str, BaseNode],
                        node_add: Union[str, BaseNode],
                        in_out_map: dict = None):
        pass

    def save_new_model(self, new_mod_path: str,
                       nodes_save: Union[Set, List] = None):
        pass


class BaseRunner(object):
    __metaclass__ = ABCMeta

    def infer(self, mod: BaseGraph, out_names: List[str],
              feed_dict=None) -> List[np.ndarray]:
        pass
