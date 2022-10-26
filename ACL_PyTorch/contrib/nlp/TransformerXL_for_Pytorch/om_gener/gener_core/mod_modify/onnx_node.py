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
from typing import Union, List, Tuple
import logging
import numpy as np
from onnx import NodeProto
from onnx import TensorProto
from onnx import ValueInfoProto
from onnx import AttributeProto
from onnx import TensorShapeProto
from onnx import helper
from onnx import numpy_helper
import numpy as np
import logging

from .interface import BaseNode
from .interface import AttrType
from ..mod_uti.log_uti import mlog


DATA_TYPE_MAP = {
    1: "float32", "float32": 1,
    2: "uint8", "uint8": 2,
    3: "int8", "int8": 3,
    4: "uint16", "uint16": 4,
    5: "int16", "int16": 5,
    6: "int32", "int32": 6,
    7: "int64", "int64": 7,
    8: "string", "string": 8,
    9: "bool", "bool": 9,
    10: "float16", "float16": 10,
    11: "double", "double": 11,
    12: "uint32", "uint32": 12,
    13: "uint64", "uint64": 13,
    14: "complex64", "complex64": 14,
    15: "complex128", "complex128": 15,
    16: "bfloat16", "bfloat16": 16
}

EXCLUDE_NODE_ATTR_TYPE = (AttrType.BOOL, AttrType.DTYPE, AttrType.SHAPE)
INPUT_TYPE = "Placeholder"
INIT_TYPE = "Initializer"
NULL_TYPE = "NULL"


def _tensor_proto_to_numpy(t_proto: TensorProto) -> np.ndarray:
    if not isinstance(t_proto, TensorProto):
        raise TypeError("not tensor proto, type: {}".format(type(t_proto)))
    return numpy_helper.to_array(t_proto)


def _tensor_proto_from_numpy(value, name: str) -> TensorProto:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError("tensor proto need list/tuple/ndarray, not {}".format(
            type(value)))
    src_array = np.array(value)
    return numpy_helper.from_array(src_array, name)


def _shape_proto_from_list(value: Union[list, tuple]) -> TensorShapeProto:
    if not isinstance(value, (list, tuple)):
        raise TypeError("shape need list/tuple, not {}".format(value))

    res = TensorShapeProto()
    for cur_v in value:
        dim = TensorShapeProto.Dimension()
        dim.dim_value = cur_v
        res.dim.append(dim)
    return res


class OXNode(BaseNode):
    """
    node class for onnx model
    """
    def __init__(self, node: Union[NodeProto, ValueInfoProto, TensorProto]):
        """
        ValueInfoProto for input, TensorProto for initializer
        """
        if not isinstance(node, (NodeProto, ValueInfoProto, TensorProto)):
            raise TypeError("need NodeProto, not {}".format(type(node)))
        self._node = node

    @property
    def node(self):
        """
        get original onnx xxproto
        """
        return self._node

    @property
    def op_type(self) -> str:
        """
        INPUT_TYPE for compatible with Tensorflowz
        """
        if isinstance(self._node, NodeProto):
            return self._node.op_type
        if isinstance(self._node, ValueInfoProto):
            return INPUT_TYPE
        else:
            return INIT_TYPE

    def set_op(self, op_type: str):
        if not isinstance(op_type, str):
            raise TypeError("need str, not {}".format(type(op_type)))
        if isinstance(self._node, NodeProto):
            self._node.op_type = op_type
        else:
            raise RuntimeError("{} can't set op".format(type(self._node)))

    @property
    def name(self) -> str:
        return self._node.name

    def set_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("need str, not {}".format(type(name)))
        self._node.name = name

    def get_attr(self, name: str, attr_type: AttrType):
        res = self.get_multi_attr({name: attr_type})
        return res.get(name)

    def get_multi_attr(self, name_types: dict) -> dict:
        # input:  name as key, AttrType as value
        # output: name as key, value as value
        if not isinstance(name_types, dict):
            raise TypeError("need dict, not {}".format(type(name_types)))

        attr_map = {}
        if isinstance(self._node, NodeProto):
            parse_func = self._parase_node_attr
            for attr in self._node.attribute:
                attr_map[attr.name] = attr
        elif isinstance(self._node, ValueInfoProto):
            parse_func = self._parse_value_info_attr
        else:
            parse_func = self._parse_tensor_attr

        res = {}
        for name, a_type in name_types.items():
            if not isinstance(a_type, AttrType):
                raise TypeError("invalid {} type {}".format(name, a_type))
            res[name] = parse_func(attr_map, name, a_type)
        return res

    def set_attr(self, attr_infos: dict):
        # name as key, (AttrType, value) as value
        if not isinstance(attr_infos, dict):
            raise TypeError("need dict, not {}".format(type(attr_infos)))

        attr_map = {}
        if isinstance(self._node, NodeProto):
            set_func = self._set_node_attr
            for attr in self._node.attribute:
                attr_map[attr.name] = attr
        elif isinstance(self._node, ValueInfoProto):
            set_func = self._set_value_info_attr
        else:
            set_func = self._set_tensor_attr

        for name, info in attr_infos.items():
            if not isinstance(name, str):
                raise TypeError("invalid attr name type {}".format(type(name)))
            if not isinstance(info, (tuple, list)):
                raise TypeError("invalid attr info: {}".format(type(info)))
            if len(info) != 2:
                raise ValueError("invalid attr info: {}".format(info))
            if not isinstance(info[0], AttrType):
                raise TypeError("invalid {} attr: {}".format(name, info[0]))
            set_func(attr_map, name, info[0], info[1])
        return True

    @property
    def shape(self) -> List[int]:           # for placeholder or input node
        if self.op_type == INPUT_TYPE:
            return self.get_attr("shape", AttrType.SHAPE)
        else:
            raise RuntimeError("only input have shape attr")

    def set_shape(self, shape: List[int]):
        if self.op_type == INPUT_TYPE:
            return self.set_attr({"shape": (AttrType.SHAPE, shape)})
        else:
            raise RuntimeError("only input have shape attr")

    def get_rand_tensor(self, rand_range) -> np.ndarray:
        shape = self.shape
        dtype = self.get_attr("dtype", AttrType.DTYPE)
        return (rand_range * np.random.random(shape)).astype(dtype)

    @property
    def const_value(self) -> np.ndarray:    # for const node
        if self.op_type in (INIT_TYPE, "Constant"):
            return self.get_attr("value", AttrType.TENSOR)
        else:
            raise RuntimeError("only const node have const value. {}".format(
                self.op_type))

    def set_const_value(self, value: np.array):
        if self.op_type in (INIT_TYPE, "Constant"):
            return self.set_attr({"value": (AttrType.TENSOR, value)})
        else:
            raise RuntimeError("only const node have const value. {}".format(
                self.op_type))

    @property
    def input_name(self) -> List[str]:
        if self.op_type in (INIT_TYPE, INPUT_TYPE):
            return []
        else:
            return list(self._node.input)

    @property
    def out_name(self) -> List[str]:
        if self.op_type in (INIT_TYPE, INPUT_TYPE):
            return [self.name]
        else:
            return list(self._node.output)

    def set_out_name(self, out_num=1):
        if self.op_type in (INIT_TYPE, INPUT_TYPE):
            return
        if out_num <= 0:
            raise ValueError("out_num:{} should > 0".format(out_num))

        cur_out_num = len(self._node.output)
        for idx in range(0, cur_out_num):
            self._node.output[idx] = "{}_{}".format(self._node.name, idx)
        for idx in range(cur_out_num, out_num):
            self._node.output.append("{}_{}".format(self._node.name, idx))

    def copy_input_from_node(self, node):
        if len(node.input_name) < len(self.input_name):
            mlog("{}'s input num {} < {}'s in num{}".format(node.name,
                len(node.input_name), self.name, len(self.input_name)),
                 level=logging.WARNING)
        for idx, in_name in enumerate(node.input_name):
            if idx < len(self._node.input):
                self._node.input[idx] = in_name
            else:
                self._node.input.append(in_name)

    def set_input_node(self, start_index: int, node_all: Union[List, Tuple],
                       node_all_output_index: List[int] = None):
        if self.op_type in (INIT_TYPE, INPUT_TYPE):
            raise RuntimeError("{} can't set input".format(self.op_type))
        if start_index > len(self._node.input):
            raise RuntimeError("start index {} error".format(start_index))
        out_idxs = node_all_output_index
        if out_idxs is None:
            out_idxs = [0 for node in node_all]
        if len(out_idxs) != len(node_all):
            raise RuntimeError("node idx num not match. {}, {}".format(
                len(node_all), len(out_idxs)))

        for idx, in_node in enumerate(node_all):
            if isinstance(in_node, self.__class__):
                in_name = in_node.out_name[out_idxs[idx]]
            elif isinstance(in_node, str):
                in_name = in_node
            else:
                raise RuntimeError("{} input {}: type {} not supported".format(
                    self.name, idx, type(in_node)))

            if start_index < len(self._node.input):
                self._node.input[start_index] = in_name
            else:
                self._node.input.append(in_name)
            start_index += 1

    def _parse_value_info_attr(self, attr_map, attr_name, attr_type: AttrType):
        if attr_type == AttrType.SHAPE:
            dims = self._node.type.tensor_type.shape.dim
            return [dims[i].dim_value for i in range(len(dims))]
        if attr_type == AttrType.DTYPE:
            return DATA_TYPE_MAP.get(self._node.type.tensor_type.elem_type)

        raise RuntimeError("ValueInfoProto only support shape and dtype, \
            {} not supported".format(attr_type))

    def _parse_tensor_attr(self, attr_map, attr_name, attr_type: AttrType):
        if attr_type == AttrType.SHAPE:
            return list(self._node.dims) if (len(self._node.dims) > 0) else [1]
        if attr_type == AttrType.DTYPE:
            return DATA_TYPE_MAP.get(self._node.data_type)
        if attr_type == AttrType.TENSOR:
            return _tensor_proto_to_numpy(self._node)

        raise RuntimeError("TensorProto only support shape, dtype and tensor \
            {} not supported".format(attr_type))

    def _parase_node_attr(self, attr_map, attr_name, attr_type: AttrType):
        if attr_type.base_type() in EXCLUDE_NODE_ATTR_TYPE:
            raise ValueError("NodeProto not support {} attr".format(attr_type))

        cur_attr = attr_map.get(attr_name)
        if cur_attr is None:
            raise RuntimeError("{} don't have attr {}".format(
                self._node.name, attr_name))
        res = helper.get_attribute_value(cur_attr)

        if attr_type == AttrType.TENSOR:
            return _tensor_proto_to_numpy(res)
        if attr_type == AttrType.LIST_TENSOR:
            return [_tensor_proto_to_numpy(x) for x in res]
        return res

    def _set_value_info_attr(self, attr_map, name, attr_type: AttrType, value):
        if attr_type == AttrType.SHAPE:
            shape_proto = _shape_proto_from_list(value)
            self._node.type.tensor_type.shape.CopyFrom(shape_proto)
        elif attr_type == AttrType.DTYPE:
            self._node.type.tensor_type.elem_type = DATA_TYPE_MAP.get(value)
        else:
            raise RuntimeError("ValueInfoProto only support shape and dtype, \
                {} not supported".format(attr_type))

    def _set_tensor_attr(self, attr_map, name, attr_type: AttrType, value):
        if attr_type == AttrType.TENSOR:
            tensor_proto = _tensor_proto_from_numpy(value, self._node.name)
            self._node.CopyFrom(tensor_proto)
        else:
            raise RuntimeError("TensorProto only support set tensor, \
                {} not supported".format(attr_type))

    def _set_node_attr(self, attr_map, name, attr_type: AttrType, value):
        if attr_type.base_type() in EXCLUDE_NODE_ATTR_TYPE:
            raise ValueError("NodeProto not support {} attr".format(attr_type))
        if attr_type == AttrType.TENSOR:
            attr_v = _tensor_proto_from_numpy(value, "")
        elif attr_type == AttrType.LIST_TENSOR:
            attr_v = [_tensor_proto_from_numpy(v, "v_{}".format(i))
                for i,v in enumerate(value)]
        elif attr_type == AttrType.FLOAT:
            attr_v = float(value)
        elif attr_type == AttrType.LIST_FLOAT:
            attr_v = [float(v) for v in value]
        else:
            attr_v = value
        res_attr = helper.make_attribute(name, attr_v)
        if name in attr_map:
            attr_map.get(name).CopyFrom(res_attr)
        else:
            self._node.attribute.append(res_attr)
