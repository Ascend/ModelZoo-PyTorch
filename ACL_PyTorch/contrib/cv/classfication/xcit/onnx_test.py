# Copyright 2021 Huawei Technologies Co., Ltd
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

import sys
import time
import onnx
from onnx import TensorProto
from onnxmltools.utils.float16_converter import convert_float_to_float16
import onnxruntime
import torch
import numpy as np
import struct


def onnx_align(model, onnx_path, batch_size, fp16):
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    torch_out = model(x)
    onnx_model = onnx.load(onnx_path)
    if fp16:
        x = x.type(torch.float16)
        onnx_model = convert_float_to_float16(onnx_model)
        onnx.save_model(onnx_model, onnx_path)
        return
    onnx.save_model(onnx_model, onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_path)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def speed_test(onnx_path, batch_size):
    inputs = torch.randn(batch_size, 3, 224, 224, requires_grad=True).type(torch.float16)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
    print(onnxruntime.get_device())
    total_time = 0.0
    for i in range(100):
        start_time = time.time()
        ort_session.run(None, ort_inputs)
        time_in = time.time() - start_time
        # print("cost time:",time_in)
        total_time = total_time + time_in
    print("average time:", total_time / 100)
    print(" FPS:", batch_size * 100 / float(total_time))


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def sel_onnx(model, err):
    graph = model.graph
    nodes = graph.node
    for i in range(len(nodes)):
        if nodes[i].output[0] == err:
            # print(nodes[i].input)
            # nodes[i].input[0] = new_node.output[0]
            return i


def onnx_modify(model, err):
    graph = model.graph
    nodes = graph.node
    for i in range(len(nodes)):
        if nodes[i].name == err:
            new_node = onnx.helper.make_node('Cast',
                                             name=err + '_cast',
                                             inputs=['input'],
                                             outputs=['output_for_' + err],
                                             to=getattr(TensorProto, 'INT32'),
                                             )
            # nodes[i].input[0] = new_node.output[0]
            pre_node_index = sel_onnx(model, nodes[i].input[0])
            new_node.input[0] = nodes[pre_node_index].output[0]
            nodes[i].input[0] = new_node.output[0]
            print(new_node)
            model.graph.node.insert(i, new_node)
            return


def onnx_clip_repair(onnx_path):
    assert 'Not finished yet.'
    sub_const_node = onnx.helper.make_tensor(name='max_clip_value',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=[1],
                                             vals=[127.5])
    onnx_model.graph.initializer.append(sub_const_node)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(onnx_model.graph.initializer)


def add_cast(errs, onnx_model, save_name):
    for err in errs:
        onnx_modify(onnx_model, err)
    onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, save_name)


def remove_int64(onnx_model, save_name):
    # print(onnx_model)
    # #for node in onnx_model.input.node:
    # #print(node.data_type)
    for node in onnx_model.graph.node:
        if node.op_type == 'Constant':
            if node.attribute[0].t.data_type == 7:  # int64
                print("find int64 at", node.name)
                node.attribute[0].t.data_type = getattr(TensorProto, 'INT32')
                print(node.attribute[0].t.raw_data)
                print(struct.unpack('<L', node.attribute[0].t.raw_data[:4]))
    onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, save_name)


if __name__ == '__main__':
    if sys.argv[1] == 'speed':
        onnx_path = sys.argv[2]
        batch_size = int(sys.argv[3])
        speed_test(onnx_path, batch_size)
