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
import onnx
import argparse
import numpy as np
from gener_core.mod_modify.interface import AttrType as AT
from gener_core.mod_modify.onnx_graph import OXGraph


def create_sigmoid_fit(mod, io_map, sigmoid_node, i, **const_value):
    conv_node = mod.get_node(sigmoid_node.input_name[0])
    mul_node = mod.get_node(io_map.get(sigmoid_node.name)[0])

    pow_node1 = mod.add_new_node(f"Pow1_n{i}", "Pow")
    pow_node1.set_input_node(0, [conv_node, const_value['exp_node1']])
    pow_node2 = mod.add_new_node(f"Pow2_n{i}", "Pow")
    pow_node2.set_input_node(0, [conv_node, const_value['exp_node1']])
    pow_node3 = mod.add_new_node(f"Pow3_n{i}", "Pow")
    pow_node3.set_input_node(0, [conv_node, const_value['exp_node2']])
    
    mul_node1 = mod.add_new_node(f"Mul1_n{i}", "Mul")
    mul_node1.set_input_node(0, [conv_node, const_value['m_node1']])
    mul_node2 = mod.add_new_node(f"Mul2_n{i}", "Mul")
    mul_node2.set_input_node(0, [pow_node1, const_value['m_node2']])
    mul_node3 = mod.add_new_node(f"Mul3_n{i}", "Mul")
    mul_node3.set_input_node(0, [pow_node2, const_value['m_node3']])
    mul_node4 = mod.add_new_node(f"Mul4_n{i}", "Mul")
    mul_node4.set_input_node(0, [mul_node3, pow_node3])

    sub_node = mod.add_new_node(f"Sub_n{i}", "Sub")
    sub_node.set_input_node(0, [mul_node1, mul_node2])
    
    add_node1 = mod.add_new_node(f"Add1_n{i}", "Add")
    add_node1.set_input_node(0, [sub_node, mul_node4])
    add_node2 = mod.add_new_node(f"Add2_n{i}", "Add")
    add_node2.set_input_node(0, [add_node1, const_value['a_node1']])

    clip_node = mod.add_new_node(f"Clip_n{i}", "Clip")
    clip_node.set_input_node(0, [add_node2, const_value['c_node1'], const_value['c_node2']])

    mul_node.set_input_node(0, [conv_node, clip_node])


def delete_sigmoid_fit(mod, io_map, sigmoid_node):
    dequant_node = mod.get_node(sigmoid_node.input_name[0])
    tmp_node = mod.get_node(io_map.get(dequant_node.name)[1])
    i = tmp_node.name.split('n')[-1]
    mul_node = mod.get_node(io_map.get(f"Clip_n{i}")[0])
    mul_node.set_input_node(0, [dequant_node, sigmoid_node])
    mod.node_remove([f"Pow1_n{i}", f"Pow2_n{i}", f"Pow3_n{i}"])
    mod.node_remove([f"Mul1_n{i}", f"Mul2_n{i}", f"Mul3_n{i}", f"Mul4_n{i}"])
    mod.node_remove([f"Sub_n{i}", f"Add1_n{i}", f"Add2_n{i}", f"Clip_n{i}"])


def create_const(mod):
    # exp
    exp_node1 = mod.add_const_node("const_exp1", np.array(3, np.float32))
    exp_node2 = mod.add_const_node("const_exp2", np.array(2, np.float32))
    # mul
    m_node1 = mod.add_const_node("const_mul1", np.array(0.229270815, np.float32))
    m_node2 = mod.add_const_node("const_mul2", np.array(0.0102459298, np.float32))
    m_node3 = mod.add_const_node("const_mul3", np.array(0.000207697530, np.float32))
    # add
    a_node1 = mod.add_const_node("const_add", np.array(0.5, np.float32))
    # clip
    c_node1 = mod.add_const_node("const_clip1", np.array(0, np.float32))
    c_node2 = mod.add_const_node("const_clip2", np.array(1, np.float32))
    
    const_value = {'exp_node1': exp_node1,
                'exp_node2': exp_node2,
                'm_node1': m_node1,
                'm_node2': m_node2,
                'm_node3': m_node3,
                'a_node1': a_node1,
                'c_node1': c_node1,
                'c_node2': c_node2}
    
    return const_value


def delete_const(mod):
    mod.node_remove(["const_exp1", "const_exp2", "const_mul1", "const_mul2", "const_mul3", 
                     "const_add", "const_clip1", "const_clip2"])


def pre_amct(input_onnx, output_onnx):
    mod = OXGraph(input_onnx)
    io_map = mod.get_net_in_out_map()

    const_value = create_const(mod)
    sigmoid_nodes = mod.get_nodes_by_optype("Sigmoid")
    for i, s_node in enumerate(sigmoid_nodes):
        create_sigmoid_fit(mod, io_map, s_node, i, **const_value)

    mod.save_new_model(output_onnx)


def after_amct(input_onnx, output_onnx):
    mod = OXGraph(input_onnx)
    io_map = mod.get_net_in_out_map()

    delete_const(mod)
    sigmoid_nodes = mod.get_nodes_by_optype("Sigmoid")
    for i, s_node in enumerate(sigmoid_nodes):
        delete_sigmoid_fit(mod, io_map, s_node)
    
    mod.save_new_model(output_onnx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("modify yolov5 onnx model for calibrating scale")
    parser.add_argument('--input-onnx', type=str, default='./yolov5s.onnx', help='input_onnx path')
    parser.add_argument('--output-onnx', type=str, default='./yolov5s.onnx', help='output_onnx path')
    parser.add_argument('--mode', type=str, default='pre_amct', help='run time')
    args = parser.parse_args()
    if args.mode == 'pre_amct':
        pre_amct(args.input_onnx, args.output_onnx)
    elif args.mode == 'after_amct':
        after_amct(args.input_onnx, args.output_onnx)
