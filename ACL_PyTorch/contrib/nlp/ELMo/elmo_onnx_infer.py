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

import onnxruntime as ort
import onnx
import numpy as np
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='elmo_sim.onnx')
parser.add_argument('--loop', default=20, type=int)
opt = parser.parse_args()

onnx = onnx.load_model(opt.model)
sess = ort.InferenceSession(onnx.SerializeToString())
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name


start = datetime.datetime.now()
for i in range(opt.loop):
    data = np.random.randint(low=0, high=1, size=(1, 8, 50), dtype='int32')
    outputs = sess.run([output_name], {input_name: data})
    print('{0} finished'.format(i))
end = datetime.datetime.now()
gap = float((end - start).total_seconds())

sample = gap / opt.loop
print('mean:{0}ms, throughput:{1}'.format(sample * 1000, 1000 / (sample * 1000)))
