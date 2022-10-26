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
from typing import List, Set, Tuple, Dict, Union
import traceback
import numpy as np
import tensorflow as tf

from .interface import BaseGraph
from .interface import AttrType
from .tf_node import TfNode


def catch_error(fun_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as error_get:
                print(error_get)
                traceback.print_exc()
                raise RuntimeError("[ERROR][TfGraphNodes][{}]input params {} does not match".format(fun_name, args))
        return wrapper
    return decorator


class TfGraphNodes(BaseGraph):

    """
    Attributes:
        model_name: 读取模型的名称
        nodes: {node_name: (node_index, node)...}
        nodes_name: [node_name...]
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._graph = tf.compat.v1.get_default_graph()
        self._graph_def = self._import_model()
        self._save_nodes_all = set()
        self._if_save = False

    @property
    def nodes(self) -> Dict:
        """
        :return: {name: tuple(索引位置, NodeDef)}
        """
        nodes = {}
        for index_1, node in enumerate(self._graph_def.node):
            nodes[node.name] = (index_1, node)
        return nodes

    @property
    def nodes_name(self) -> List[str]:
        """
        :return: List[node_name]
        """
        names = []
        for node in self._graph_def.node:
            names.append(node.name)
        return names

    @catch_error("import_model")
    def _import_model(self):
        """
        获取对应的图形和图形的序列化表示方式
        :return: 默认图形，序列化表示方式
        :raise: RuntimeError(读取图时发生错误)
        """
        graph_def = self._graph.as_graph_def()
        graph_def.ParseFromString(tf.compat.v1.gfile.FastGFile(self.model_name, "rb").read())
        return graph_def

    def _name_decide(self, name: str):
        if name in self.nodes_name:
            raise RuntimeError("{} node name already exists".format(name))

    @catch_error("add_const_node")
    def add_const_node(self, name: str, value: np.ndarray, if_array: bool = True) -> TfNode:
        if not if_array:
            value = value[0]
        return self.add_new_node(name, "Const",
                                 {"value": (AttrType.TENSOR, value),
                                  "dtype": (AttrType.DTYPE, value.dtype)})

    @catch_error("add_placeholder_node")
    def add_placeholder_node(self, name: str, data_type: str, shape: Union[List, Tuple]) -> TfNode:
        """
        添加placeholder节点
        :param name: 节点名称
        :param data_type 数据格式
        :param shape: 输入格式
        :return: 新生成的const_node
        """
        node_def = self._graph_def.node.add()
        tf_node = TfNode(self, node_def)
        tf_node.set_name(name)
        tf_node.set_op("Placeholder")
        tf_node.set_attr({"dtype": (AttrType.DTYPE, data_type),
                          "shape": (AttrType.SHAPE, shape)})
        return tf_node

    @catch_error("add_new_node")
    def add_new_node(self, name, op_type: str, attrs: dict = None,
                     out_num=1) -> TfNode:
        """
        only add, no connection
        :attrs: name as key, (AttrType, value) as value
        """
        node_def = self._graph_def.node.add()
        tf_node = TfNode(self, node_def)
        tf_node.set_name(name)
        tf_node.set_op(op_type)
        tf_node.set_attr(attrs)
        return tf_node

    @catch_error("getitem")
    def __getitem__(self, index: int) -> TfNode:
        """
        获得index处的节点
        :param index: 索引位置
        :return: Node
        """
        node = self._graph_def.node[index]
        node = TfNode(self, node)
        return node

    @catch_error("get_node")
    def get_node(self, name: Union[str, TfNode]) -> TfNode:
        """
        根据name返回TfNode
        :param name: node name
        :return: TfNode
        """
        if isinstance(name, str):
            for node in self._graph_def.node:
                if name == node.name:
                    return TfNode(self, node)
            raise RuntimeError("{} node_name doesn't exists".format(name))
        else:
            return name

    @catch_error("get_nodes_forward_node")
    def get_nodes_forward_node(self, node_i: Union[str, TfNode],
                               input_index: Union[List[int], Tuple[int]] = None,
                               if_self: bool = True, end_nodes: List[Union[str, TfNode]] = None,
                               if_end_nodes: List[bool] = None) -> Set[str]:
        """
        获得图中该节点，包括该节点在内的，以及指定输入在内一直到最上方的节点名称
        :param node_i: 指定节点
        :param input_index: 指定节点输入
        :param if_self: 是否包含自己
        :param end_nodes: 结束节点，不能是node_i
        :param if_end_nodes: 是否包含结束节点，默认包含
        :return: 图中该节点上方的节点的名称
        """
        if end_nodes is None:
            end_nodes_str = []
        else:
            end_nodes_str = []
            for node in end_nodes:
                if isinstance(node, TfNode):
                    end_nodes_str.append(node.name)
                else:
                    end_nodes_str.append(node)
        if if_end_nodes is None:
            if_end_nodes = [True] * len(end_nodes_str)
        node_i = self.get_node(node_i)
        input_node_all = set()
        if if_self:
            input_node_all.add(node_i.name)
        node_input = self.get_input_node(node_i)
        node_temp_all = []
        if input_index is None:
            input_index = [i for i in range(len(node_input))]
        for index_1 in input_index:
            node_temp_all.append(node_input[index_1])
        while node_temp_all:
            node_temp = node_temp_all.pop()
            if node_temp.name in input_node_all:
                continue
            elif node_temp.name in end_nodes_str:
                if if_end_nodes[end_nodes_str.index(node_temp.name)]:
                    input_node_all.add(node_temp.name)
            else:
                input_node_all.add(node_temp.name)
                node_temp_all.extend(self.get_input_node(node_temp))
        return input_node_all

    @catch_error("get_nodes_behind_node")
    def get_nodes_behind_node(self, name: Union[str, TfNode], if_self: bool = False) -> Set[str]:
        """
        获得图中以该节点为输入的所有节点
        :param name: 指定节点
        :param if_self: 是否包含自己
        :return: 图中该节点下方的节点的名称
        """
        node_all = set(self.nodes_name)
        return node_all - self.get_nodes_forward_node(name, if_self=not if_self)

    @catch_error("get_input")
    def get_input(self, node: Union[str, TfNode]) -> List[str]:
        """
        获得所有输入节点的名字和索引，形如name:index
        :return: List[str]
        """
        node = self.get_node(node)
        return node.input_name_index

    @catch_error("get_input_node_name")
    def get_input_node_name(self, node: Union[str, TfNode]) -> List[str]:
        """
        获得所有输入节点的名字
        :return: List[str]
        """
        node = self.get_node(node)
        return node.input_name

    @catch_error("get_input_node")
    def get_input_node(self, node: Union[str, TfNode]) -> List[TfNode]:
        """
        获得所有输入节点
        :return: List[TfNode]
        """
        node = self.get_node(node)
        input_node = []
        for node_name in node.input_name:
            input_node.append(self.get_node(node_name))
        return input_node

    @catch_error("get_output")
    def get_output(self, node_input: Union[str, TfNode]) -> Dict[int, List[Tuple]]:
        """
        获取输出节点的详细信息
        :param node_input:
        :return: dict{输出节点输出索引: list(tuple(输入节点索引，TfNode))}
        """
        output_node = {}
        if isinstance(node_input, TfNode):
            node_input = node_input.name
        for node in self._graph_def.node:
            tf_node = TfNode(self, node)
            name_all = tf_node.input_name
            input_info_all = tf_node.input_name_index
            for index_input, node_name in enumerate(name_all):
                if node_input == node_name:
                    index_output = input_info_all[index_input].split(":")[1]
                    output_node.setdefault(int(index_output), []).append((index_input, tf_node))
        return output_node

    @catch_error("get_output_node_name")
    def get_output_node_name(self, node_input: Union[str, TfNode]) -> List[str]:
        """
        获得所有输出节点的名字
        :return: List[str]
        """
        if isinstance(node_input, TfNode):
            node_input = node_input.name
        output_node_name = []
        for node in self._graph_def.node:
            tf_node = TfNode(self, node)
            if node_input in tf_node.input_name:
                output_node_name.append(node.name)
        return output_node_name

    @catch_error("get_output_node")
    def get_output_node(self, node_input: Union[str, TfNode]) -> List[TfNode]:
        """
        获得所有输出节点
        :return: List[TfNode]
        """
        if isinstance(node_input, TfNode):
            node_input = node_input.name
        output_node = []
        for node in self._graph_def.node:
            tf_node = TfNode(self, node)
            if node_input in tf_node.input_name:
                output_node.append(TfNode(self, node))
        return output_node

    @catch_error("save_new_model")
    def save_new_model(self, model_name: str):
        """
        保存一个新的pb
        :param model_name: 新pb的名称
        :return: None
        """
        new_model = tf.compat.v1.GraphDef()
        assert(new_model != self._graph_def)
        for node_temp in self._graph_def.node:
            if self._if_save and node_temp.name not in self._save_nodes_all:
                continue
            node_new = new_model.node.add()
            node_new.CopyFrom(node_temp)
        with open(model_name, "wb") as file_open:
            file_open.write(new_model.SerializeToString())

    @catch_error("node_move")
    def node_move(self, node_move_name: Union[str, TfNode], node_name_forward: Union[str, TfNode],
                  forward: bool = True, if_remove: bool = True) -> TfNode:
        """
        把node_move_name在GraphDef上的顺序移到node_name_forward的前面
        :param node_move_name: 需要移动的节点
        :param node_name_forward: 目标节点位置
        :param forward: 目标节点的前面还是后面
        :param if_remove: 是否删除原来位置的节点
        :return: None
        """
        node_all = self.nodes
        if isinstance(node_move_name, TfNode):
            node_move_name = node_move_name.name
        if isinstance(node_name_forward, TfNode):
            node_name_forward = node_name_forward.name
        node_move = node_all[node_move_name][1]
        index_node = node_all[node_name_forward][0]
        if not forward:
            index_node += 1
        self._graph_def.node.insert(index_node, node_move)
        if if_remove:
            nodes_name = self.nodes_name
            node_new_name = node_move.name
            while node_new_name in nodes_name:
                node_new_name += "/remove"
            node_move.name = node_new_name
            self._graph_def.node.remove(node_move)
        return TfNode(self, self._graph_def.node[index_node])

    @catch_error("node_remove")
    def node_remove(self, nodes_name: Union[Set[Union[str, TfNode]], List[Union[str, TfNode]]]):
        """
        把node_move_name在GraphDef上的顺序移到node_name_forward的前面
        :param nodes_name: 需要删除的节点的集合
        :return: None
        """
        for node in nodes_name:
            if isinstance(node, str):
                node = self.get_node(node)
            self._graph_def.node.remove(node.node)

    @catch_error("node_save")
    def node_save(self, nodes_save: Union[Set[Union[str, TfNode]], List[Union[str, TfNode]]]):
        """
        保存节点的集合，一旦调用这个函数，save_new_model只会保留这个函数要求保存的节点
        :param nodes_save: 保存节点
        :return: None
        """

        self._if_save = True
        for node in nodes_save:
            if isinstance(node, TfNode):
                node = node.name
            self._save_nodes_all.add(node)

    @catch_error("node_replace")
    def node_replace(self, node_original: Union[str, TfNode], node_new: Union[str, TfNode],
                     input_node: List[Union[str, TfNode]] = None,
                     input_node_index: int = 0,
                     new_node_output: Dict[int, int] = None,
                     if_remove: bool = True):
        """
        替换节点，需要注意，在指定新节点输入时，需要保证输入全部输入，及从第0个输入到最后一个输入全部指定好
        :param node_original: 原节点
        :param node_new: 新节点
        :param input_node: 新节点的所有输入节点，不指定默认获取原节点所有输入节点
        :param input_node_index: 输入节点的起始索引
        :param new_node_output: 新节点第几个输出来替换原节点输出
        :param if_remove: 是否删除原节点
        :return:
        """
        node_original = self.get_node(node_original)
        node_new = self.get_node(node_new)
        output_node_all = self.get_output(node_original)
        if input_node is None:
            tar_input_node = self.get_input_node_name(node_original)
        else:
            tar_input_node = []
            for node in input_node:
                if isinstance(node, TfNode):
                    tar_input_node.append(node.name)
                elif isinstance(node, str):
                    tar_input_node.append(node)
                else:
                    raise RuntimeError("unsupported input {}".format(node))
        node_new.set_input_node(input_node_index, tar_input_node)

        for output_index, node_info in output_node_all.items():
            for index_input, tf_node in node_info:
                if new_node_output is not None and output_index in new_node_output.keys():
                    output_index = new_node_output[output_index]
                tf_node.set_input_node(index_input, [node_new], [output_index])
        if if_remove:
            self.node_remove([node_original, ])

    @catch_error("node_add_forward")
    def node_add_forward(self, node_original: Union[str, TfNode],
                         node_add: Union[str, TfNode],
                         node_original_index: int = 0,
                         node_add_input_index: int = 0,
                         node_add_output_index: int = 0):
        """
        在一个节点前面添加一个节点
        :param node_original: 原节点
        :param node_add: 添加的节点
        :param node_original_index: 原节点第几个输入前加节点
        :param node_add_input_index: 添加的节点第几个输入接收原节点的输入
        :param node_add_output_index: 新节点第几个输出来替换原节点输出
        :return:
        """
        node_original = self.get_node(node_original)
        node_add = self.get_node(node_add)
        node_input = node_original.input_original
        node_add.set_input_node(node_add_input_index, [node_input[node_original_index]])
        node_original.set_input_node(node_original_index, [node_add], [node_add_output_index])

    @catch_error("node_add_behind")
    def node_add_behind(self, node_original: Union[str, TfNode],
                        node_add: Union[str, TfNode],
                        node_original_index: int = 0,
                        node_add_input_index: int = 0,
                        node_add_output_index: int = 0):
        """
        在一个节点后面添加一个节点
        :param node_original: 原节点
        :param node_add: 添加的节点
        :param node_original_index: 原节点第几个输出
        :param node_add_input_index: 添加的节点第几个输入接收原节点的输入
        :param node_add_output_index: 新节点第几个输出来替换原节点输出
        :return:
        """
        node_original = self.get_node(node_original)
        output_info = self.get_output(node_original)
        node_add = self.get_node(node_add)
        node_add.set_input_node(node_add_input_index, [node_original], [node_original_index])
        for _, node_info in output_info.items():
            for input_index, tf_node in node_info:
                tf_node.set_input_node(input_index, [node_add], [node_add_output_index])

    @catch_error("node_remove_connect")
    def node_remove_connect(self, node_remove: Union[str, TfNode], input_index: int = 0, if_remove: bool = True):
        """
        删除现节点，并把该节点输入节点和输出节点相连
        :param node_remove: 删除节点
        :param input_index: 原节点那个输入接到缺失的地方
        :param if_remove: 是否删除现节点
        :return:
        """
        node_remove = self.get_node(node_remove)
        input_name = node_remove.input_original[input_index]
        output_info = self.get_output(node_remove)
        for _, node_info in output_info.items():
            for input_index, tf_node in node_info:
                tf_node.set_input_node(input_index, [input_name])
        if if_remove:
            self.node_remove([node_remove])

    @catch_error("node_copy")
    def node_copy(self, node_copy: Union[str, TfNode], name: str = None) -> TfNode:
        node_copy = self.get_node(node_copy)
        self._graph_def.node.append(node_copy.node)
        node_new = TfNode(self, self._graph_def.node[-1])
        if name is None:
            self._name_decide(node_copy.name)
        else:
            node_new.set_name(name)
        return node_new

    @catch_error("get_net_input_nodes")
    def get_net_input_nodes(self) -> List[TfNode]:
        return self.get_nodes_by_optype("Placeholder")

    @catch_error("get_nodes_by_optype")
    def get_nodes_by_optype(self, tar_optype) -> List[TfNode]:
        tar_nodes = []
        for node in self._graph_def.node:
            if (node.op == tar_optype):
                tar_nodes.append(TfNode(self, node))
        return tar_nodes

    @catch_error("get_subgraph")
    def get_subgraph_name(self, subgraph_func, *args) -> Set[str]:
        subgraph_name_set = set()
        for node in self._graph_def.node:
            tf_node = TfNode(self, node)
            if subgraph_func(tf_node, *args):
                subgraph_name_set.add(node.name)
        return subgraph_name_set

    @catch_error("get_subgraph_input_node")
    def get_subgraph_input_nodename(self, subgraph_func, *args) -> Set[str]:
        subgraph_node_set = self.get_subgraph_name(subgraph_func, *args)
        node_input_set = set()
        for node_name in subgraph_node_set:
            node_input_list = self.get_input_node(node_name)
            for node_input in node_input_list:
                node_input_set.add(node_input.name)
        return node_input_set - subgraph_node_set

    @catch_error("get_subgraph_output_node")
    def get_subgraph_output_nodename(self, subgraph_func, *args) -> Set[str]:
        subgraph_node_set = self.get_subgraph_name(subgraph_func, *args)
        node_output_set = set()
        for node_name in subgraph_node_set:
            node_output_list = self.get_output_node_name(node_name)
            for node_output in node_output_list:
                node_output_set.add(node_output)
        return node_output_set - subgraph_node_set

    def get_net_in_out_map(self):
        res = {}
        for node in self._graph_def.node:
            cur_out_name = node.name
            for in_name in node.input:
                in_name = in_name.split(":")[0]
                if (in_name not in res):
                    res[in_name] = [cur_out_name]
                else:
                    res[in_name].append(cur_out_name)
        return res

    def node_remove_from(self, start_node: Union[str, TfNode]):
        in_out_map = self.get_net_in_out_map()
        start_node = self.get_node(start_node)
        start_name = start_node.name
        candi_names = start_node.input_name
        names_to_remove = set([start_name])
        for tar_name in candi_names:
            cur_out = set(in_out_map[tar_name])
            if (cur_out.issubset(names_to_remove)):
                names_to_remove.add(tar_name)
                tar_node = self.get_node(tar_name)
                candi_names.extend(tar_node.input_name)
        self.node_remove(names_to_remove)
        return names_to_remove

    def get_net_output_nodes(self) -> List[TfNode]:
        res = []
        in_out_map = self.get_net_in_out_map()
        for node in self._graph_def.node:
            if (node.name not in in_out_map):
                res.append(TfNode(self, node))
        return res
