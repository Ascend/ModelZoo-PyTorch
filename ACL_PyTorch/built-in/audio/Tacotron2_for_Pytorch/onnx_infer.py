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

import onnxruntime as rt
import numpy as np

def onnxruntime_init(model):
    sess = rt.InferenceSession(model)
    input_name = []
    for n in sess.get_inputs():
        input_name.append(n.name)
    output_name = []
    for n in sess.get_outputs():
        output_name.append(n.name)
    return sess, input_name, output_name


def onnxruntime_run(sess, input_name, output_name, input_data):
    res_buff = []
    succ = True

    res = sess.run(None, {input_name[i]: input_data[i] for i in range(len(input_name))})
    for i, x in enumerate(res):
        out = np.array(x)
        res_buff.append(out)
    
    return res_buff, succ


class Waveglow():
    def __init__(self, waveglow):
        self.sess, self.input_name, self.output_name = onnxruntime_init(waveglow)

    def infer(self, mel):
        mel = np.array(mel)
        mel_size = mel.shape[2]
        batch_size = mel.shape[0]
        stride = 256
        n_group = 8
        z_size = mel_size * stride
        z_size = z_size // n_group
        z = np.random.randn(batch_size, n_group, z_size)

        z = z.astype(np.float32)
        mel = mel.astype(np.float32)

        waveglow_output, _ = onnxruntime_run(self.sess, self.input_name, self.output_name, [mel, z])
        waveglow_output = np.reshape(waveglow_output, (batch_size, -1))
        return waveglow_output


