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
from typing import List, Union, Tuple
import traceback

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework.node_def_pb2 import NodeDef

from .interface import BaseNode
from .interface import AttrType


ATTR_FIELD_MAP = {AttrType.BOOL: "b",
                  AttrType.BYTE: "s",
                  AttrType.FLOAT: "f",
                  AttrType.INT: "i",
                  AttrType.DTYPE: "type",
                  AttrType.TENSOR: "tensor",
                  AttrType.SHAPE: "shape"}


DATA_TYPE_NUM = {
    1: "float32",
    2: "float64",
    3: "int32",
    4: "uint8",
    5: "int16",
    6: "int8",
    7: "string",
    8: "complex64",
    9: "int64",
    10: "bool",
}


def catch_error(fun_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as error_get:
                print(error_get)
                traceback.print_exc()
                raise RuntimeError("[ERROR][TfNode][{}]input params {} does not match".format(fun_name, args))
        return wrapper
    return decorator


class TfNode(BaseNode):
    """
    节点类，主要是修改节点的操作
    """

    def __init__(self, graph_nodes, node: NodeDef):
        """
        :param graph_nodes: GraphNodes
        :param node: tf.compat.v1.NodeDef
        """
        self.graph_nodes = graph_nodes
        self._node = node

    @property
    def node(self) -> NodeDef:
        return self._node

    @property
    def name(self) -> str:
        return self._node.name

    @property
    def op_type(self) -> str:
        return self._node.op

    @property
    def const_value(self) -> np.ndarray:
        if self.op_type == "Const":
            return tensor_util.MakeNdarray(self.node.attr["value"].tensor)
        else:
            raise RuntimeError("{} is not const node".format(self.name))

    def set_const_value(self, value: np.array):
        if (self.op_type == "Const"):
            self.set_attr({"value": (AttrType.TENSOR, value)})
        else:
            raise RuntimeError("unsupported op {} for set const value".format(
                self.op_type))

    @property
    def input_name(self) -> List[str]:
        """
        获得所有输入节点的名字
        :return: List[str]
        """
        input_node_name = []
        for node_name in self._node.input:
            node_info = node_name.split(":")
            if len(node_info) > 2:
                raise RuntimeError("{} node name cannot contain ':'".format(node_name))
            if node_info[0].startswith("^"):
                input_node_name.append(node_info[0][1:])
            else:
                input_node_name.append(node_info[0])
        return input_node_name

    @property
    def input_name_index(self) -> List[str]:
        """
        获得所有输入节点的名字和索引，形如name:index
        :return: List[str]
        """
        input_node = []
        for node_name in self._node.input:
            node_info = node_name.split(":")
            if len(node_info) == 1:
                node_name = node_name + ":0"
            elif len(node_info) > 2:
                raise RuntimeError("{} node name cannot contain ':'".format(node_name))
            input_node.append(node_name)
        return input_node

    @property
    def input_original(self) -> List[str]:
        """
        获得所有输入节点的名字和索引，和本来一致
        :return: List[str]
        """
        input_node = []
        for node_name in self._node.input:
            input_node.append(node_name)
        return input_node

    @catch_error("set_name")
    def set_name(self, name: str):
        """
        设置名称，不要和该图已有名称相同
        :param name: 修改的名称
        """
        if name in self.graph_nodes.nodes_name:
            raise RuntimeError("{} node name already exists".format(name))
        self._node.name = name

    @catch_error("set_op")
    def set_op(self, op_name: str):
        """
        设置op, 及该节点表示的算子的类型
        :param op_name: 修改的op
        """
        self._node.op = op_name

    @catch_error("set_value")
    def set_value(self, value: np.ndarray, if_array: bool = True):
        # decrpeted interface. use set_attr
        if (not if_array):
            value = value[0]
        self.set_attr({"value": (AttrType.TENSOR, value),
                       "dtype": (AttrType.DTYPE, value.dtype)})

    @catch_error("set_attr_value")
    def set_attr_value(self, attr_name: str, value, data_type: str, if_list=False, encoding="utf-8"):
        """
        decrpeted interface. use set_attr
        :param data_type: 支持{"string", "int32", "float32", "bool"}
        :param if_list: 是否是列表
        :param encoding: 如果传入string，默认编码格式utf-8
        """
        type_map = {"string": AttrType.BYTE, "int32": AttrType.INT,
                    "float32": AttrType.FLOAT, "bool": AttrType.BOOL}
        attr_type = type_map.get(data_type)
        if (attr_type is None):
            raise RuntimeError("unsupported data_type:{}".format(data_type))
        if (if_list):
            attr_type = attr_type.list_type()
        if (data_type == "string"):
            value = bytes(value, encoding=encoding)

        self.set_attr({attr_name: (attr_type, value)})

    @catch_error("set_input_node")
    def set_input_node(self, start_index: int, node_all: Union[List, Tuple], output_index: List[int] = None):
        """
        设置节点的输入
        :param start_index: 插入的索引
        :param node_all: 添加的节点
        :param output_index: 节点的第几个输出，为None默认第0个
        :raise 索引位置出错
        """
        input_length = len(self._node.input)
        if start_index > input_length:
            raise RuntimeError("{} index error".format(start_index))
        for index_1, node in enumerate(node_all):
            if isinstance(node, self.__class__):
                name = node.name
            elif isinstance(node, str):
                name = node
            else:
                raise RuntimeError("{} data type does not support".format(node_all))
            if output_index is not None and output_index[index_1] != 0:
                name = name + ":"+str(output_index[index_1])
            if start_index < len(self._node.input):
                self._node.input[start_index] = name
            else:
                self._node.input.append(name)
            start_index += 1

    @catch_error("remove_input")
    def remove_input(self, remove_node: Union[int, str]):
        if isinstance(remove_node, int):
            self._node.input.pop(remove_node)
        elif isinstance(remove_node, str):
            self._node.input.remove(remove_node)
        else:
            raise RuntimeError("{} does not support".format(type(remove_node)))

    @staticmethod
    def _create_attr(attr_type:AttrType, value):
        if (not isinstance(attr_type, AttrType)):
            raise RuntimeError("need AttrType has {}".format(attr_type))

        tar_attr = tf.compat.v1.AttrValue.ListValue() \
            if attr_type.is_list() else tf.compat.v1.AttrValue()

        base_type = attr_type.base_type()
        tar_field = ATTR_FIELD_MAP.get(base_type)
        if (base_type in (AttrType.BOOL, AttrType.BYTE,
                          AttrType.FLOAT, AttrType.INT)):
            if (attr_type == AttrType.LIST_BOOL):
                tar_attr = tf.compat.v1.AttrValue.ListValue(b=value)
            elif (attr_type == AttrType.LIST_BYTE):
                tar_attr = tf.compat.v1.AttrValue.ListValue(s=value)
            elif (attr_type == AttrType.LIST_FLOAT):
                tar_attr = tf.compat.v1.AttrValue.ListValue(f=value)
            elif (attr_type == AttrType.LIST_INT):
                tar_attr = tf.compat.v1.AttrValue.ListValue(i=value)
            else:
                setattr(tar_attr, tar_field, value)
        elif (base_type == AttrType.SHAPE):
            shape = tf.TensorShape(value).as_proto()
            # Assignment not allowed to field "shape" in protocol message obj.
            tar_attr = tf.compat.v1.AttrValue.ListValue(shape=shape) \
            if attr_type.is_list() else tf.compat.v1.AttrValue(shape=shape)
        elif (base_type == AttrType.DTYPE):
            setattr(tar_attr, tar_field, tf.as_dtype(value).as_datatype_enum)
        elif (base_type == AttrType.TENSOR):
            # Assignment not allowed to field "tensor" in protocol message obj.
            tensor = tensor_util.make_tensor_proto(value)
            tar_attr = tf.compat.v1.AttrValue.ListValue(tensor=tensor) \
            if attr_type.is_list() else tf.compat.v1.AttrValue(tensor=tensor)
        else:
            raise RuntimeError("unsupported attr type:{}".format(attr_type))

        if attr_type.is_list():
            return tf.compat.v1.AttrValue(list=tar_attr)
        else:
            return tar_attr

    @staticmethod
    def _parse_attr(attr_v:tf.compat.v1.AttrValue, attr_type:AttrType):
        if (not isinstance(attr_type, AttrType)):
            raise RuntimeError("need AttrType has {}".format(attr_type))

        tar_attr = attr_v.list if attr_type.is_list() else attr_v
        base_type = attr_type.base_type()
        tar_field = ATTR_FIELD_MAP.get(base_type)
        if (base_type in (AttrType.BOOL, AttrType.BYTE,
                          AttrType.FLOAT, AttrType.INT)):
            return getattr(tar_attr, tar_field)
        elif (base_type == AttrType.SHAPE):
            shape_proto = getattr(tar_attr, tar_field)
            return tf.TensorShape(shape_proto).as_list()
        elif (base_type == AttrType.DTYPE):
            return DATA_TYPE_NUM.get(getattr(tar_attr, tar_field))
        elif (base_type == AttrType.TENSOR):
            tensor = getattr(tar_attr, tar_field)
            return tensor_util.MakeNdarray(tensor)
        else:
            raise RuntimeError("unsupported attr type:{}".format(attr_type))

    def get_attr(self, name: str, attr_type:AttrType):
        res = self.get_multi_attr({name: attr_type})
        return res.get(name)

    def get_multi_attr(self, name_types: dict) -> dict:
        res = {}
        for name, attr_type in name_types.items():
            cur_v = self._parse_attr(self._node.attr[name], attr_type)
            res[name] = cur_v
        return res

    def set_attr(self, attr_infos: dict):
        for name, info in attr_infos.items():
            if (len(info) != 2):
                raise RuntimeError("attr info (type, value). {}".format(info))

            tar_attr = self._create_attr(info[0], info[1])
            self._node.attr[name].CopyFrom(tar_attr)

    @property
    def shape(self) -> list:
        if (self.op_type == "Placeholder"):
            return tensor_util.TensorShapeProtoToList(
                self._node.attr["shape"].shape)
        else:
            raise RuntimeError("unsupported op {} for shape".format(
                self.op_type))

    def set_shape(self, shape: List[int]):
        if (self.op_type == "Placeholder"):
            self.set_attr({"shape": (AttrType.SHAPE, shape)})
        else:
            raise RuntimeError("unsupported op {} for shape".format(
                self.op_type))

    def get_rand_tensor(self, rand_range) -> np.ndarray:
        # Placeholder return by shape, other op return None
        if (self.op_type == "Placeholder"):
            shape = self.shape
            dtype = self.get_attr("type", AttrType.DTYPE)
            return (rand_range * np.random.random(shape)).astype(dtype)
        else:
            return None

    def __repr__(self):
        return self._node.__repr__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.graph_nodes == other.graph_nodes and self.name == other.name
        else:
            raise RuntimeError("[ERROR][GraphNode][__eq__][{}] can't compare".format(self.name))
