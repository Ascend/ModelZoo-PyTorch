# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from pyacl.acl_infer import AclNet, init_acl, release_acl
import acl
import numpy as np
import onnxruntime

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b)/(a_norm * b_norm)
    return cos

if __name__ == '__main__':
    DEVICE = 0
    init_acl(DEVICE)
    chunk_xs = np.random.random((64, 67, 80)).astype("float32")
    chunk_lens = np.array([600]*64).astype("int32")
    offset = np.array([0]*64).reshape((64, 1))
    att_cache = np.random.random((64, 12, 4, 64, 128)).astype("float32")
    cnn_cache = np.random.random((64, 12, 256, 7)).astype("float32")
    cache_mask = np.random.random((64, 1, 64)).astype("float32")  
    net = AclNet(model_path="online_encoder.om", device_id=DEVICE)
    output_data, exe_time = net([chunk_xs, chunk_lens, offset, att_cache, cnn_cache, cache_mask])
    y = output_data[0].flatten()
    onnx_model = onnxruntime.InferenceSession("./onnx/online_encoder.onnx")
    inputs = {
        onnx_model.get_inputs()[0].name: chunk_xs,
        onnx_model.get_inputs()[1].name: chunk_lens,
        onnx_model.get_inputs()[2].name: offset,
        onnx_model.get_inputs()[3].name: att_cache,
        onnx_model.get_inputs()[4].name: cnn_cache,
        onnx_model.get_inputs()[5].name: cache_mask,

    }
    onnx_output = onnx_model.run(None, inputs)
    onnx_y = onnx_output[0].flatten()
    cos_1 = cos_sim(y, onnx_y)
    print("acc: ", cos_1)
    del net
    release_acl(DEVICE)
