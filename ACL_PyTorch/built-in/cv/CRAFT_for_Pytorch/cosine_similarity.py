# Copyright 2020 Huawei Technologies Co., Ltd
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
    input_data = np.random.random((1, 3, 640, 640)).astype("float32")
    net = AclNet(model_path="craft.om", device_id=DEVICE)
    output_data, exe_time = net([input_data])
    y = output_data[0].flatten()
    feature = output_data[1].flatten()
    onnx_model = onnxruntime.InferenceSession("craft.onnx")
    inputs = {
        onnx_model.get_inputs()[0].name: input_data,
    }
    onnx_output = onnx_model.run(None, inputs)
    onnx_y = onnx_output[0].flatten()
    onnx_feature = onnx_output[1].flatten()
    cos_1 = cos_sim(y, onnx_y)
    cos_2 = cos_sim(feature, onnx_feature)
    print("cosine_1:", cos_1)
    print("cosine_2:", cos_2)
    del net
    release_acl(DEVICE)
