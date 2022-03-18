# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

'''
环境：
    python==3.8.5
    onnx==1.8.1
    onnxruntime==1.7.0
    skl2onnx==1.8.0
    numpy==1.19.5
'''

import os
import sys
import onnx
import copy
import time
import shutil
import numpy as np
import onnxruntime

from enum import IntEnum
from onnx import NodeProto
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, NoReturn
from onnx.numpy_helper import from_array, to_array
from onnx.onnx_ml_pb2 import TensorProto, ValueInfoProto, AttributeProto
from onnx.helper import make_attribute, make_node, make_graph, make_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs, select_model_inputs_outputs, save_onnx_model

# 修改递归深度限制
sys.setrecursionlimit(100000)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> enum OXDataType
# onnx类型枚举值
class OXDataType(IntEnum):
    float32 = 1
    uint8 = 2
    int8 = 3
    uint16 = 4
    int16 = 5
    int32 = 6
    int64 = 7
    string = 8
    bool = 9
    float16 = 10
    double = 11
    uint32 = 12
    uint64 = 13
    complex64 = 14
    complex128 = 15
    bfloat16 = 16


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> calss GV
# 全局变量，GV = global variable
class GV:
    # onnx和numpy数据类型映射字典
    ONNX_2_NUMPY_DATATYPE_DICT = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        4: np.uint16,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        9: np.bool_,
        10: np.float16,
        11: np.float64,
        12: np.uint32,
        13: np.uint64,
        14: np.complex64,
        15: np.complex128,
        np.float32: 1,
        np.uint8: 2,
        np.int8: 3,
        np.uint16: 4,
        np.int16: 5,
        np.int32: 6,
        np.int64: 7,
        np.bool_: 9,
        np.float16: 10,
        np.float64: 11,
        np.uint32: 12,
        np.uint64: 13,
        np.complex64: 14,
        np.complex128: 15,
        'tensor(float)': np.float32,
        'tensor(uint8)': np.uint8,
        'tensor(int8)': np.int8,
        'tensor(uint16)': np.uint16,
        'tensor(int16)': np.int16,
        'tensor(int32)': np.int32,
        'tensor(int64)': np.int64,
        'tensor(bool)': np.bool_,
        'tensor(float16)': np.float16,
        'tensor(double)': np.float64,
        'tensor(uint32)': np.uint32,
        'tensor(uint64)': np.uint64,
    }

    # initializer，node索引字典（实现快速查找）
    OXINITIALIZER_DICT = {}
    OXNODE_DICT = {}


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> calss OXInitializer
class OXInitializer:
    '''
    If you print a Initializer variable in the terminal, you will get something like this, and you can modify it
    directly.
    dims: 1
    data_type: 6
    name: '4239'
    raw_data: '\000\000\000\00'

    @dims: google.protobuf.pyext._message.RepeatedScalarContainer
    @data_type: int
    @name: str
    @raw_data: bytes
    '''

    def __init__(self, initializer: TensorProto):
        self._initializer = initializer

    def __str__(self):
        ndarray = to_array(self._initializer)
        msg = 'name: ' + str(self._initializer.name) + '\n' + \
              'dims: ' + str(self._initializer.dims) + '\n' + \
              'data_type: ' + str(self._initializer.data_type) + '\n' + \
              'dtype: ' + str(ndarray.dtype) + '\n' + \
              'shape: ' + str(ndarray.shape) + '\n' + \
              'ndarray:\n' + str(ndarray)
        return msg

    def get_initializer(self) -> TensorProto:
        return self._initializer

    def get_name(self) -> str:
        '''
        获取initializer的名字
        '''

        return self._initializer.name

    def set_name(self, new_name) -> NoReturn:
        '''
        设置/修改initializer的名字
        '''

        old_name = self._initializer.name
        self._initializer.name = new_name
        GV.OXINITIALIZER_DICT[new_name] = GV.OXINITIALIZER_DICT[old_name]
        GV.OXINITIALIZER_DICT.pop(old_name)

    def get_data_type(self) -> int:
        '''
        获取initializer的数据类型
        '''

        return self._initializer.data_type

    def set_data_type(self, ox_data_type: OXDataType) -> NoReturn:
        '''
        设置/修改initializer的数据类型
        '''

        ndarray = to_array(self._initializer).astype(GV.ONNX_2_NUMPY_DATATYPE_DICT[int(ox_data_type)])
        self._initializer.raw_data = ndarray.tobytes()
        self._initializer.data_type = int(ox_data_type)

    def get_data(self) -> np.ndarray:
        '''
        获取initializer的数据
        '''

        return to_array(self._initializer)

    def set_data(self, ndarray: np.ndarray) -> NoReturn:
        '''
        设置/修改initializer的数据
        '''

        self._initializer.raw_data = ndarray.tobytes()
        self._initializer.data_type = GV.ONNX_2_NUMPY_DATATYPE_DICT[eval('np.' + str(ndarray.dtype))]
        _clear_list(self._initializer.dims)
        _extend_list(self._initializer.dims, ndarray.shape)

    def save_data(self, file_path: str) -> NoReturn:
        '''
        保存initializer的数据
        '''

        np.save(file_path, to_array(self._initializer))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> calss OXNode
class OXNode:
    '''
    If you print a NodeProto variable in the terminal, you will get something like this, and you can modify it directly.
    input: '494'
    input: 'fc.weight'
    input: 'fc.bias'
    output: 'class'
    name: 'Gemm_121'
    op_type: 'Gemm'
    attribute {
      name: 'alpha'
      f: 1.0
      type: FLOAT
    }
    attribute {
      name: 'beta'
      f: 1.0
      type: FLOAT
    }
    attribute {
      name: 'transB'
      i: 1
      type: INT
    }

    @input: google.protobuf.pyext._message.RepeatedScalarContainer
    @output: google.protobuf.pyext._message.RepeatedScalarContainer
    @name: str
    @op_type: str
    @attribute: google.protobuf.pyext._message.RepeatedCompositeContainer
    '''

    def __init__(self, node: NodeProto):
        self._node = node

    def __str__(self):
        return str(self._node)

    def get_node(self) -> NodeProto:
        return self._node

    @property
    def input(self):  # -> google.protobuf.pyext._message.RepeatedScalarContainer
        '''
        获取节点的输入列表
        '''

        return self._node.input

    @property
    def output(self):  # -> google.protobuf.pyext._message.RepeatedScalarContainer
        '''
        获取节点的输出列表
        '''

        return self._node.output

    def get_name(self) -> str:
        '''
        获取节点的名字
        '''

        return self._node.name

    def set_name(self, new_name) -> NoReturn:
        '''
        设置/修改节点的名字
        '''

        old_name = self._node.name
        self._node.name = new_name
        GV.OXNODE_DICT[new_name] = GV.OXNODE_DICT[old_name]
        GV.OXNODE_DICT.pop(old_name)

    def get_op_type(self) -> int:
        '''
        获取节点的类型
        '''

        return self._node.op_type

    def set_op_type(self, op_type) -> NoReturn:
        '''
        设置/修改节点的类型
        '''

        self._node.op_type = op_type

    def get_attribute(self):  # -> google.protobuf.pyext._message.RepeatedCompositeContainer
        '''
        获取节点属性
        '''

        return self._node.attribute

    def set_attribute(self, attr_name: str, attr_value: Any) -> AttributeProto:
        '''
        设置/修改节点属性

        Args:
            attr_name: 属性名字
            attr_value: 属性值

        Returns: 修改后的属性
        '''

        # 构造新attr
        new_attr = make_attribute(attr_name, attr_value)

        # 删除旧的
        for attr in self._node.attribute:
            if attr.name == attr_name:
                self._node.attribute.remove(attr)
                break

        # 添加新的
        self._node.attribute.append(new_attr)

        return new_attr

    def add_attribute(self, attr_name: str, attr_value: Any) -> AttributeProto:
        '''
        给节点增加新属性

        Args:
            attr_name: 属性名字
            attr_value: 属性值

        Returns: 新增的属性
        '''

        # 构造新attr
        new_attr = make_attribute(attr_name, attr_value)

        # 增加
        self._node.attribute.append(new_attr)

        return new_attr

    def remove_attribute(self, attr_name: str) -> AttributeProto:
        '''
        删除节点的某个属性

        Args:
            attr_name: 属性名字
            attr_value: 属性值

        Returns: 被删除的属性
        '''

        for attr in self._node.attribute:
            if attr.name == attr_name:
                removed_attr = attr
                self._node.attribute.remove(attr)
                break

        return removed_attr


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> calss OXGraph
class OXGraph:
    def __init__(self, model_path: str):
        print('[INFO] Start initializing.')
        start_time = datetime.now()

        self._model_path = model_path
        self._model = onnx.load_model(model_path)
        self._graph = self._model.graph
        self._initializer = self._graph.initializer
        self._node = self._graph.node
        self._input_tensor_2_oxnode_dict = {}
        self._output_tensor_2_oxnode_dict = {}

        # initializer
        for initializer in self._initializer:
            GV.OXINITIALIZER_DICT[initializer.name] = OXInitializer(initializer)

        # node
        for idx, node in enumerate(self._node):
            oxnode = OXNode(node)
            GV.OXNODE_DICT[node.name] = oxnode

        # 创建tensor_2_oxnode字典
        self._update_tensor_2_oxnode_dict(
            self._input_tensor_2_oxnode_dict,
            self._output_tensor_2_oxnode_dict,
        )

        # 获取所有tensor信息
        try:
            self._all_tensor_info = self.get_all_tensor_info()
        except:
            os.remove(os.path.join(os.path.dirname(self._model_path), 'temp.onnx'))
            print('[WARNING] There are custom operators in the model, '
                  'and these functions are not available: get_input_tensor_info()、get_output_tensor_info()、'
                  'get_all_tensor_info()、infer_shape()、dump_all_node_data()、trunc_model().')

        # 屏蔽check_model
        def check_model(model):
            pass

        onnx.checker.check_model = check_model

        end_time = datetime.now()
        cost_time = (end_time - start_time).seconds
        print('[INFO] Initialization completed! Cost {} seconds.'.format(cost_time))

    def __str__(self):
        return str(self._model)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Initializer相关函数
    def get_oxinitializer_by_name(self, oxinitializer_name: str, can_return_none: bool = False) -> OXInitializer:
        '''
        根据initializer的名字获取OXInitializer
        '''

        if oxinitializer_name not in GV.OXINITIALIZER_DICT:
            if can_return_none is True:
                return None
            else:
                raise RuntimeError('[ERROR] {} not found.'.format(oxinitializer_name))
        return GV.OXINITIALIZER_DICT[oxinitializer_name]

    def add_initializer(self, initializer_name: str, ndarray: np.ndarray) -> OXInitializer:
        '''
        向模型中新增一个initializer

        Args:
            initializer_name: initializer的名字
            ndarray: initializer的数据

        Returns: 新增的OXInitializer
        '''

        if initializer_name in GV.OXINITIALIZER_DICT:
            raise RuntimeError(
                '[ERROR] {} has already exists in the model, please use a different name!'.format(initializer_name))

        initializer = from_array(ndarray, initializer_name)
        self._initializer.append(initializer)  # 这里是复制，而不是引用，id已经变了
        initializer = self._initializer[-1]
        oxinitializer = OXInitializer(initializer)
        GV.OXINITIALIZER_DICT[initializer_name] = oxinitializer

        return oxinitializer

    def remove_initializer(self, initializer_name: str) -> OXInitializer:
        '''
        从模型中删除指定的initializer

        Args:
            initializer_name: initializer的名字

        Returns: 删除的OXInitializer
        '''

        oxinitializer = self.get_oxinitializer_by_name(initializer_name)
        GV.OXINITIALIZER_DICT.pop(initializer_name)
        self._initializer.remove(oxinitializer.get_initializer())

        return oxinitializer

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Node相关函数
    def get_oxnode_by_name(self, oxnode_name: str, can_return_none: bool = False) -> OXNode:
        '''
        根据节点名字获取OXNode
        '''

        if oxnode_name not in GV.OXNODE_DICT:
            if can_return_none is True:
                return None
            else:
                raise RuntimeError('[ERROR] {} not found.'.format(oxnode_name))
        return GV.OXNODE_DICT[oxnode_name]

    def get_oxnode_by_op_type(self, op_type: str) -> List[OXNode]:
        '''
        根据节点类型获取OXNode
        '''

        res = set()
        for oxnode in GV.OXNODE_DICT.values():
            if oxnode.get_op_type() == op_type:
                res.add(oxnode)
        return list(res)

    def get_oxnode_whose_input_contain_this(self, input_name: str) -> List[OXNode]:
        '''
        遍历所有OXNode，获取输入包含`input_name`的那些OXNode
        '''

        res = set()
        for oxnode in GV.OXNODE_DICT.values():
            for oxinput_name in oxnode.input:
                if oxinput_name == input_name:
                    res.add(oxnode)
                    break
        return list(res)

    def get_oxnode_whose_output_contain_this(self, output_name: str) -> List[OXNode]:
        '''
        遍历所有OXNode，获取输出包含`output_name`的那些OXNode
        '''

        res = set()
        for oxnode in GV.OXNODE_DICT.values():
            for oxoutput_name in oxnode.output:
                if oxoutput_name == output_name:
                    res.add(oxnode)
                    break
        return list(res)

    def get_previous_oxnode(self, oxnode_name: str) -> List[OXNode]:
        '''
        获取一个节点的前驱节点
        '''

        res = set()
        inputs = self.get_oxnode_by_name(oxnode_name).input
        for input in inputs:
            oxnode_set = self._output_tensor_2_oxnode_dict.get(input)
            if oxnode_set is not None:
                res.update(oxnode_set)
        return list(res)

    def get_next_oxnode(self, oxnode_name: str) -> List[OXNode]:
        '''
        获取一个节点的后继节点
        '''

        res = set()
        outputs = self.get_oxnode_by_name(oxnode_name).output
        for output in outputs:
            oxnode_set = self._input_tensor_2_oxnode_dict.get(output)
            if oxnode_set is not None:
                res.update(oxnode_set)
        return list(res)

    def insert_node(self, bef_node_info_list: List[Dict], aft_node_info_list: List[Dict], op_type: str, op_name: str,
                    **attributes: Dict) -> OXNode:
        '''
        向模型中插入新节点，并自动连边，注意和`add_node`的区别

        限制：无法涵盖所有场景，若结果不符合预期，请用`add_node`函数，并手动指定连边关系。

        Args:
            bef_node_info_list：参见README.md用例
            aft_node_info_list：参见README.md用例
            op_type：节点的类型
            op_name：节点的名字
            attributes：节点的属性

        Returns: 插入的OXNode
        '''

        # 校验插入的节点是否已经存在
        if op_name in GV.OXNODE_DICT:
            raise RuntimeError(
                '[ERROR] {} has already exists in the model, please use a different name!'.format(op_name))

        # 解析信息
        bef_node_info_list, aft_node_info_list = self._parse_insert_node_info(bef_node_info_list, aft_node_info_list)

        # 插入节点
        # + 构造新节点的输入
        new_node_input = []
        for bef_node_info in bef_node_info_list:
            oxnode = self.get_oxnode_by_name(bef_node_info['bef_node_name'], True)
            if oxnode is None:  # 说明此节点是模型的输入节点
                new_node_input.append(bef_node_info['bef_node_name'])
            else:
                for idx in bef_node_info['link_output_idx']:
                    if oxnode.output[idx] in self.get_output_tensor_info().keys():  # 说明此节点紧接模型的输出节点
                        oxnode.output[idx] = oxnode.get_name() + '_m_' + str(idx)
                    new_node_input.append(oxnode.output[idx])

        # + 构造新节点的输出
        new_node_output = [op_name + '_0']

        # + 构造新节点
        insert_oxnode = self.add_node(op_type=op_type,
                                      op_name=op_name,
                                      inputs=new_node_input,
                                      outputs=new_node_output,
                                      **attributes)

        # 和后继节点连边
        for aft_node_info in aft_node_info_list:
            oxnode = self.get_oxnode_by_name(aft_node_info['aft_node_name'], True)
            if oxnode is None:  # 说明此节点是模型的输出节点
                if len(aft_node_info_list) != 1:
                    raise RuntimeError('[ERROR] Please check aft_node_info_list!')

                # 修改insert_oxnode的输出为模型的输出节点
                insert_oxnode.output[0] = aft_node_info['aft_node_name']
            else:
                for idx in aft_node_info['link_input_idx']:
                    oxnode.input[idx] = new_node_output[0]

        # 更新tensor_2_oxnode字典
        self._update_tensor_2_oxnode_dict(
            self._input_tensor_2_oxnode_dict,
            self._output_tensor_2_oxnode_dict,
        )

        return insert_oxnode

    def add_node(self, op_type: str, op_name: str, inputs: List[str], outputs: List[str], **attributes: Dict) -> OXNode:
        '''
        向模型中增加新节点，不会自动连边，注意和`insert_node`的区别

        Args:
            op_type：节点的类型
            op_name：节点的名字
            inputs：节点的输入
            outputs：节点的输出
            attributes：节点的属性

        Returns: 新增的OXNode
        '''

        if op_name in GV.OXNODE_DICT:
            raise RuntimeError(
                '[ERROR] {} has already exists in the model, please use a different name!'.format(op_name))

        new_node = make_node(op_type=op_type, name=op_name, inputs=inputs, outputs=outputs, **attributes)
        self._node.append(new_node)  # 这里复制，而不是用引用，id已经变了
        new_node = self._node[-1]
        new_oxnode = OXNode(new_node)
        GV.OXNODE_DICT[new_oxnode.get_name()] = new_oxnode

        # 更新tensor_2_oxnode字典
        self._update_tensor_2_oxnode_dict(
            self._input_tensor_2_oxnode_dict,
            self._output_tensor_2_oxnode_dict,
        )

        return new_oxnode

    def remove_node(self, node_name: str, auto_link: bool = True) -> OXNode:
        '''
        从模型中删除节点

        限制：若开启自动连边，则删除的节点必须只有一个前驱节点，否则需要手动连边。若结果不符合预期，也需要自己手动连边。

        Args:
            node_name：要删除的节点名字
            auto_link：是否自动连边

        Returns: 删除的OXNode
        '''

        if node_name not in GV.OXNODE_DICT:
            raise RuntimeError('[ERROR] {} not found.'.format(node_name))

        if auto_link is False:
            oxnode = self.get_oxnode_by_name(node_name)
        else:
            oxnode = self.get_oxnode_by_name(node_name)
            previous_node = self.get_previous_oxnode(node_name)
            next_node = self.get_next_oxnode(node_name)

            if len(previous_node) > 1:
                raise RuntimeError('[ERROR] Remove node can only have one previous node.')

            _clear_list(previous_node[0].output)
            _extend_list(previous_node[0].output, oxnode.output)

        # 删除节点
        GV.OXNODE_DICT.pop(node_name)
        self._node.remove(oxnode.get_node())

        # 更新tensor_2_oxnode字典
        self._update_tensor_2_oxnode_dict(
            self._input_tensor_2_oxnode_dict,
            self._output_tensor_2_oxnode_dict,
        )

        return oxnode

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 输入输出相关函数
    def get_input_tensor_info(self) -> Dict:
        '''
        获取模型输入tensor的信息
        信息包括：tensor名字、shape、类型

        Returns: {'tensor_name': {'shape': np.shape, 'dtype': np.dtype}, ...}
        '''

        session = onnxruntime.InferenceSession(self._model_path)

        input_tensor_info = {}
        for input_item in session.get_inputs():
            input_tensor_info[input_item.name] = {
                'shape': tuple(input_item.shape),
                'dtype': GV.ONNX_2_NUMPY_DATATYPE_DICT[input_item.type]
            }

        return input_tensor_info

    def get_output_tensor_info(self) -> Dict:
        '''
        获取模型输出tensor的信息
        信息包括：tensor名字、shape、类型

        Returns: {'tensor_name': {'shape': np.shape, 'dtype': np.dtype}, ...}
        '''

        session = onnxruntime.InferenceSession(self._model_path)

        output_tensor_info = {}
        for output_item in session.get_outputs():
            output_tensor_info[output_item.name] = {
                'shape': tuple(output_item.shape),
                'dtype': GV.ONNX_2_NUMPY_DATATYPE_DICT[output_item.type]
            }

        return output_tensor_info

    def get_all_tensor_info(self) -> Dict:
        '''
        获取模型中所有tensor的信息
        所有tensor包括：模型输入tensor、模型输出tensor、模型中间tensor
        信息包括：tensor名字、shape、类型

        Returns: {'tensor_name': {'shape': np.shape, 'dtype': np.dtype}, ...}
        '''

        old_onnx_model = onnx.load(self._model_path)

        output_name = []
        for name in enumerate_model_node_outputs(old_onnx_model):
            output_name.append(name)

        new_onnx_model = select_model_inputs_outputs(old_onnx_model, output_name)
        new_model_path = os.path.join(os.path.dirname(self._model_path), 'temp.onnx')
        save_onnx_model(new_onnx_model, new_model_path)

        session = onnxruntime.InferenceSession(new_model_path)
        os.remove(new_model_path)

        all_tensor_info = {}

        for input_item in session.get_inputs():
            all_tensor_info[input_item.name] = {
                'shape': tuple(input_item.shape),
                'dtype': GV.ONNX_2_NUMPY_DATATYPE_DICT[input_item.type]
            }

        for output_item in session.get_outputs():
            all_tensor_info[output_item.name] = {
                'shape': tuple(output_item.shape),
                'dtype': GV.ONNX_2_NUMPY_DATATYPE_DICT[output_item.type]
            }

        for oxinitializer in GV.OXINITIALIZER_DICT.values():
            all_tensor_info[oxinitializer.get_name()] = {
                'shape': oxinitializer.get_data().shape,
                'dtype': eval('np.' + str(oxinitializer.get_data().dtype))
            }

        return all_tensor_info

    def infer_shape(self, input_data_info_list: List[Dict]) -> Dict:
        '''
        推导模型各个算子的输出shape信息。

        用途：有些模型从onnx图中无法看出算子输出shape信息，也无法获取shape信息，通过此函数可以推导出shape信息。

        原理：用真实数据运行一遍模型，记录各个算子的输出shape信息。

        Args:
            input_data_info_list:
                [
                    {
                        'model_input_name': 'input1_name',
                        'shape': '(1, 3, 224, 224)',
                        'dtype': 'np.float32'
                    },
                    {
                        'model_input_name': 'input2_name',
                        'shape': '(1, 3, 224, 224)',
                        'dtype': 'np.float32'
                    }
                ]

        Returns: {'op_name': {'shape': np.shape, 'dtype': np.dtype}, ...}
        '''

        # 构造输入数据
        input_data_dict = {}
        for input_data_info in input_data_info_list:
            input_data_dict[input_data_info['model_input_name']] = np.full(eval(input_data_info['shape']),
                                                                           1,
                                                                           dtype=eval(input_data_info['dtype']))

        # 修改模型，增加输出节点
        old_onnx_model = onnx.load(self._model_path)
        output = []
        for out in enumerate_model_node_outputs(old_onnx_model):
            output.append(out)
        new_onnx_model = select_model_inputs_outputs(old_onnx_model, outputs=output)
        onnx_save_path = './temp.onnx'
        save_onnx_model(new_onnx_model, onnx_save_path)

        # 推理得到输出
        sess = onnxruntime.InferenceSession(onnx_save_path)
        os.remove(onnx_save_path)
        output_name = [node.name for node in sess.get_outputs()]
        res = sess.run(output_name, input_data_dict)

        # 保存数据
        infer_tensor_info = {}
        idx = 0
        for node in old_onnx_model.graph.node:
            for i in range(len(node.output)):
                infer_tensor_info[node.name] = {'output_idx': i, 'shape': res[idx].shape, 'dtype': res[idx].dtype}
                idx += 1

        return infer_tensor_info

    def dump_all_node_data(self, input_data_info_list: List[Dict], dump_data_save_path: str) -> NoReturn:
        '''
        dump模型所有节点的数据

        Args:
            input_data_info_list:
                [
                    {
                        'model_input_name': 'input1_name',
                        'npy_file_path': './0.npy',
                    },
                    {
                        'model_input_name': 'input2_name',
                        'npy_file_path': './1.npy',
                    },
                ]
            dump_data_save_path: e.g. './dump_data'

        Returns: NoReturn
        '''

        # 创建目录
        if os.path.exists(dump_data_save_path):
            shutil.rmtree(dump_data_save_path)
        os.makedirs(dump_data_save_path)

        # 修改模型，增加输出节点
        old_onnx_model = onnx.load(self._model_path)
        output = []
        for out in enumerate_model_node_outputs(old_onnx_model):
            output.append(out)
        new_onnx_model = select_model_inputs_outputs(old_onnx_model, outputs=output)
        onnx_save_path = os.path.join(dump_data_save_path, "./temp.onnx")
        save_onnx_model(new_onnx_model, onnx_save_path)

        # 获取输入数据
        input_data_dict = {}
        for input_data_info in input_data_info_list:
            input_data_dict[input_data_info['model_input_name']] = np.load(input_data_info['npy_file_path'])

        # 推理得到输出
        sess = onnxruntime.InferenceSession(onnx_save_path)
        os.remove(onnx_save_path)
        output_name = [node.name for node in sess.get_outputs()]
        res = sess.run(output_name, input_data_dict)

        # 保存数据
        idx = 0
        for node in old_onnx_model.graph.node:
            for i in range(len(node.output)):
                file_name = node.name + "." + str(i) + "." + str(round(time.time() * 1000000)) + ".npy"
                data_save_path = os.path.join(dump_data_save_path, file_name)
                np.save(data_save_path, res[idx])
                idx += 1

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 截图函数
    def extract_model(self, input_tensor_name_list: List[str], output_tensor_name_list: List[str],
                      new_model_save_path: str) -> NoReturn:
        '''
        从onnx 1.8.1开始，onnx官方提供了截图函数，此函数是对官方`onnx.utils.extract_model`函数的封装，
        以使其集成到`OXGraph`类中。另外，此函数屏蔽了`check_model`操作，使包含自定义算子的onnx提取子图后
        在保存模型时跳过检查操作，使之可以顺利保存。以下是官方`onnx.utils.extract_model`函数的说明：

        Extracts sub-model from an ONNX model.

        The sub-model is defined by the names of the input and output tensors *exactly*.

        Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
        which is defined by the input and output tensors, should not _cut through_ the
        subgraph that is connected to the _main graph_ as attributes of these operators.

        Arguments:
            input_path (string): The path to original ONNX model.
            output_path (string): The path to save the extracted ONNX model.
            input_names (list of string): The names of the input tensors that to be extracted.
            output_names (list of string): The names of the output tensors that to be extracted.
        '''

        print('[INFO] Begin to extract the model.')
        start_time = datetime.now()
        onnx.utils.extract_model(self._model_path, new_model_save_path, input_tensor_name_list, output_tensor_name_list)
        end_time = datetime.now()
        cost_time = (end_time - start_time).seconds
        print('[INFO] Extract model completed! Cost {} seconds.'.format(cost_time))

    def trunc_model(self,
                    trunc_beg_node_name_list: List[str],
                    trunc_end_node_name_list: List[str],
                    new_model_save_path: str,
                    keep_input_initializer: bool = False,
                    userdef_trunc_beg_node_info_list: List[Dict] = None) -> NoReturn:
        '''
        截取一段模型

        用途：可以用来单独验证某段网络的精度

        注意：
            从onnx 1.8.1开始，onnx官方提供了截图函数，若onnx版本>=1.8.1，请使用`extract_model`函数。
            `extract_model`函数是对官方`onnx.utils.extract_model`函数的封装，以使其集成到`OXGraph`类中。
            此`trunc_model`函数是自己写的，功能可能有缺陷，但截图速度一般来说更快，模型较大时可以对比尝试。
        '''

        print('[WARNING] 从onnx 1.8.1开始，onnx官方提供了截图函数，若onnx版本>=1.8.1，请使用`extract_model`函数。'
              '`extract_model`函数是对官方`onnx.utils.extract_model`函数的封装，以使其集成到`OXGraph`类中。'
              '此`trunc_model`函数是自己写的，功能可能有缺陷，但截图速度一般来说更快，模型较大时可以对比尝试。')

        print('[INFO] Begin to truncate the model.')
        start_time = datetime.now()

        # 修改输出节点
        new_output = []
        for elem in trunc_end_node_name_list:
            output = self.get_oxnode_by_name(elem).output
            new_output.extend(x for x in output)
        new_onnx = select_model_inputs_outputs(self._model, outputs=new_output)
        save_onnx_model(new_onnx, new_model_save_path)

        # 加载模型
        model = onnx.load_model(new_model_save_path)
        graph = model.graph
        nodes = graph.node
        initializers = graph.initializer

        # 搜索节点
        def find_trunc_beg_node(node_name):
            is_find = False
            for node in nodes:
                if node.name == node_name:
                    trunc_beg_node = node
                    is_find = True
                    break
            if is_find is True:
                return trunc_beg_node
            else:
                raise RuntimeError('[ERROR] {} not found.'.format(node_name))

        # 获取trunc_beg_node详细信息，构造一个这样的list：
        '''
        [
            {
                'trunc_beg_node': node,
                'new_input_info_list': [
                    {
                        'input_name': 'input_A',
                        'dtype': OXDataType.float32,
                        'shape': (1, 256, 56, 56),
                        'input_idx': 0
                    },
                    {
                        'input_name': 'input_B',
                        'dtype': OXDataType.float32,
                        'shape': (1, 256, 56, 56),
                        'input_idx': 1
                    }
                ]
            }
        ]
        '''
        if userdef_trunc_beg_node_info_list is None:
            trunc_beg_node_info_list = []
            initializer_name_set = set()
            initializer_name_set.update([oxinitializer.get_name() for oxinitializer in GV.OXINITIALIZER_DICT.values()])
            count = 0
            for trunc_beg_node_name in trunc_beg_node_name_list:
                trunc_beg_node = find_trunc_beg_node(trunc_beg_node_name)
                new_input_info_list = []
                for idx, input in enumerate(trunc_beg_node.input):
                    if (keep_input_initializer is True) and (input in initializer_name_set):
                        continue
                    else:
                        new_input_info = {}
                        new_input_info['input_name'] = 'new_input_' + str(count)
                        count += 1
                        new_input_info['dtype'] = GV.ONNX_2_NUMPY_DATATYPE_DICT[self._all_tensor_info[input]['dtype']]
                        new_input_info['shape'] = self._all_tensor_info[input]['shape']
                        new_input_info['input_idx'] = idx
                        new_input_info_list.append(new_input_info)
                trunc_beg_node_info = {}
                trunc_beg_node_info['trunc_beg_node'] = trunc_beg_node
                trunc_beg_node_info['new_input_info_list'] = new_input_info_list
                trunc_beg_node_info_list.append(trunc_beg_node_info)
        else:
            trunc_beg_node_info_list = userdef_trunc_beg_node_info_list

        # 构造新输入
        new_inputs = []
        for trunc_beg_node_info in trunc_beg_node_info_list:
            if userdef_trunc_beg_node_info_list is None:
                trunc_begin_node = trunc_beg_node_info['trunc_beg_node']
            else:
                trunc_begin_node = find_trunc_beg_node(trunc_beg_node_info['trunc_beg_node_name'])
            for new_input_info in trunc_beg_node_info['new_input_info_list']:
                new_input = self._make_new_input(new_input_info['input_name'], new_input_info['dtype'],
                                                 new_input_info['shape'])
                new_inputs.append(new_input)
                trunc_begin_node.input[new_input_info['input_idx']] = new_input_info['input_name']

        # 查找有用节点
        useful_node_name_set = set()
        useful_node_name_set.update(trunc_beg_node_name_list)
        useful_node_name_set.update(trunc_end_node_name_list)

        # + 正向查找
        @lru_cache()
        def find_useful_node(next_node_name_tuple):
            for next_node_name in next_node_name_tuple:
                if next_node_name not in trunc_end_node_name_list:
                    output_oxnode_list = self.get_next_oxnode(next_node_name)
                    output_oxnode_name_tuple = tuple([oxnode.get_name() for oxnode in output_oxnode_list])
                    useful_node_name_set.update(output_oxnode_name_tuple)
                    find_useful_node(output_oxnode_name_tuple)

        # + 反向查找
        @lru_cache()
        def find_useful_node_reverse(next_node_name_tuple):
            for next_node_name in next_node_name_tuple:
                if next_node_name not in trunc_beg_node_name_list:
                    input_oxnode_list = self.get_previous_oxnode(next_node_name)
                    input_oxnode_name_tuple = tuple([oxnode.get_name() for oxnode in input_oxnode_list])
                    useful_node_name_set.update(input_oxnode_name_tuple)
                    find_useful_node_reverse(input_oxnode_name_tuple)

        # + 正向和反向都查找一遍，防止漏查
        find_useful_node(tuple(trunc_beg_node_name_list))
        find_useful_node_reverse(tuple(trunc_end_node_name_list))

        # 删除多余节点
        for node in copy.deepcopy(nodes):
            if node.name not in useful_node_name_set:
                nodes.remove(node)

        # 删除多余输入
        _clear_list(graph.input)
        _extend_list(graph.input, new_inputs)

        # 删除多余Initializer
        all_input = set()
        for node in nodes:
            all_input.update(node.input)
        for initializer in copy.deepcopy(initializers):
            if initializer.name not in all_input:
                initializers.remove(initializer)

        # 保存模型
        name = 'Extracted from {' + self._graph.name + '}'
        graph = make_graph(nodes,
                           name,
                           graph.input,
                           graph.output,
                           initializer=initializers,
                           value_info=graph.value_info)
        meta = {
            'ir_version': self._model.ir_version,
            'opset_imports': self._model.opset_import,
            'producer_name': 'OXGraph.trunc_model()',
        }
        new_mode = make_model(graph, **meta)
        onnx.save(new_mode, new_model_save_path)
        end_time = datetime.now()
        cost_time = (end_time - start_time).seconds
        print('[INFO] Truncate model completed! Cost {} seconds.'.format(cost_time))

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 保存模型
    def save_new_model(self, new_model_path) -> NoReturn:
        '''
        保存修改后的模型
        '''

        onnx.save_model(self._model, new_model_path)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 私有函数
    def _update_tensor_2_oxnode_dict(self, input_tensor_2_oxnode_dict, output_tensor_2_oxnode_dict) -> NoReturn:
        # 清空字典
        input_tensor_2_oxnode_dict.clear()
        output_tensor_2_oxnode_dict.clear()

        # 创建字典
        for oxnode in GV.OXNODE_DICT.values():
            inputs = oxnode.input
            outputs = oxnode.output
            for input in inputs:
                input_tensor_2_oxnode_dict.setdefault(input, set()).add(oxnode)
            for output in outputs:
                output_tensor_2_oxnode_dict.setdefault(output, set()).add(oxnode)

    def _make_new_input(self, new_input_name: str, ox_data_type: OXDataType, shape: tuple) -> ValueInfoProto:
        '''
        If you print the model input in the terminal, you will get something like this, and you can modify it directly.
        `dim_param` means dynamic shape.
        [name: 'image'
        type {
          tensor_type {
            elem_type: 1
            shape {
              dim {
                dim_param: '-1'
              }
              dim {
                dim_value: 3
              }
              dim {
                dim_value: 224
              }
              dim {
                dim_value: 224
              }
            }
          }
        }
        ]
        '''

        new_input = copy.deepcopy(self._graph.input[0])
        new_input.name = new_input_name
        new_input.type.tensor_type.elem_type = int(ox_data_type)

        dim_diff = len(shape) - len(new_input.type.tensor_type.shape.dim)
        if dim_diff > 0:
            for i in range(dim_diff):
                new_input.type.tensor_type.shape.dim.append(copy.deepcopy(new_input.type.tensor_type.shape.dim[0]))
        elif dim_diff < 0:
            for i in range(abs(dim_diff)):
                new_input.type.tensor_type.shape.dim.pop()

        for index in range(len(shape)):
            if isinstance(shape[index], str):
                new_input.type.tensor_type.shape.dim[index].dim_param = shape[index]
            elif shape[index] is None:
                new_input.type.tensor_type.shape.dim[index].dim_param = '-1'
                print('[WARNING] Can not infer tensor shape, set it to "-1" here, which may cause an error! '
                      'Please specify `userdef_trunc_beg_node_info_list` parameters and retry.')
            else:
                new_input.type.tensor_type.shape.dim[index].dim_value = shape[index]

        return new_input

    def _parse_insert_node_info(self, bef_node_info_list, aft_node_info_list):
        '''
        parse bef_node_info_list = ['Relu_1:0'] and aft_node_info_list = ['MaxPool_2:0'] into:

        bef_node_info_list=[{
            'bef_node_name': 'Relu_1',
            'link_output_idx': [0]
        }]

        aft_node_info_list=[{
            'aft_node_name': 'MaxPool_2',
            'link_input_idx': [0]
        }]

        默认的`:0`可以省略
        '''

        # 变量定义
        new_bef_node_info_list = []
        new_aft_node_info_list = []

        # 解析bef_node_info_list
        for bef_node_info in bef_node_info_list:
            bef_node_info_dict = {}
            info_list = bef_node_info.split(':')
            bef_node_info_dict['bef_node_name'] = info_list[0]
            if len(info_list) == 1:
                bef_node_info_dict['link_output_idx'] = [0]
            else:
                bef_node_info_dict['link_output_idx'] = [int(elem) for idx, elem in enumerate(info_list) if idx > 0]
            new_bef_node_info_list.append(bef_node_info_dict)

        # 解析aft_node_info_list
        for aft_node_info in aft_node_info_list:
            aft_node_info_dict = {}
            info_list = aft_node_info.split(':')
            aft_node_info_dict['aft_node_name'] = info_list[0]
            if len(info_list) == 1:
                aft_node_info_dict['link_input_idx'] = [0]
            else:
                aft_node_info_dict['link_input_idx'] = [int(elem) for idx, elem in enumerate(info_list) if idx > 0]
            new_aft_node_info_list.append(aft_node_info_dict)

        return new_bef_node_info_list, new_aft_node_info_list


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 公共函数
def _clear_list(list) -> NoReturn:
    '''
    清空RepeatedScalarContainer或RepeatedCompositeContainer列表
    '''

    list_len = len(list)
    for _ in range(list_len):
        list.pop()


def _extend_list(list, what_to_add) -> NoReturn:
    '''
    扩展RepeatedScalarContainer或RepeatedCompositeContainer列表
    '''

    for elem in what_to_add:
        list.append(elem)
