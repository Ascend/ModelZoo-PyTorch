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

import os
import time
import torch
import onnxruntime # WARNING: there must be onnxruntime-gpu only in the running environment !
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import datetime
import torchaudio

# step 0: set the running settings here (mainly the file path)
'''-----------------------------------------------------------------------------------'''
model_path = 'tdnn.onnx' # path of the onnx model
input_info_file_path = 'mini_librispeech_test.info' # path of the input_info file
batchsize = 16 # the tested batchsize

# path of the infer result ( actually the infer time ), create if not exists
infer_res_save_path = './gpu_result'
if not(os.path.exists(infer_res_save_path)):
    os.makedirs(infer_res_save_path)

# original 'MeasureTime' copied from acl_net.py, which is used in pyacl_infer.py
class MeasureTime():
    def __init__(self, measurements, key, cpu_run=True):
        self.measurements = measurements
        self.key = key
        self.cpu_run = cpu_run

    def __enter__(self):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter_ns()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.measurements[self.key] = time.perf_counter_ns() - self.t0
'''-----------------------------------------------------------------------------------'''

# step 1: get the input file path according to the input_info file
'''-----------------------------------------------------------------------------------'''
input_file_path = {}
with open(input_info_file_path, 'rt', encoding='utf-8') as f_info:
    line = f_info.readline()
    while line:
        line = line.rstrip('\n')
        contents = line.split()
        info = {'path': contents[1], 'shape': eval(contents[2])} 
        input_file_path.setdefault(contents[0], []).append(info)
        line = f_info.readline()
'''-----------------------------------------------------------------------------------'''

# step 2: perform infer for files listed in input_file_path
'''-----------------------------------------------------------------------------------'''
if __name__ == '__main__':
    # step 2.1: set the counters
    total_infer_time = 0
    total_infer_time_workspace = 0
    total_infer_num = 0
    workspace = 10
    measurements = {}
    key = 'per_infer_time_ns'
    dim_1 = 1800
    cpu_run = False

    # step 2.2: load the model to the onnx running session
    # WARNING: there must be onnxruntime-gpu only in the running environment!
    #          if cpu and gpu exist at the same time, it will get wrong.
    onnx_run_sess = onnxruntime.InferenceSession(model_path)

    # step 2.3: for each input file, load it and perform the infer
    for key, values in tqdm(input_file_path.items()):
        # step 2.3.1: load the input data
        inputs = []
        dims = [] # dims and dims_info is actually unused here
        for idx, value in enumerate(values):
            x = np.fromfile(value['path'], dtype=np.float32).reshape(value['shape'])
            inputs.append(x)
            dims.extend(value['shape'])
        dims_info = {'dimCount': len(dims), 'name': '', 'dims': dims}
        inputs = torch.tensor(np.array(inputs).squeeze(axis = 0))
        pad = dim_1 - inputs.shape[1]
        inputs = torch.nn.functional.pad(inputs, (0,0,0,pad,0,0), value=0).numpy()

        # step 2.3.2: perform the infer
        with MeasureTime(measurements, key, cpu_run):
            _ = onnx_run_sess.run(None, {onnx_run_sess.get_inputs()[0].name:inputs})
        total_infer_num += 1

        # step 2.3.3: save the output => pass
    
        # step 2.3.4: calculate the time
        total_infer_time += measurements[key]
        if total_infer_num > workspace:
            total_infer_time_workspace += measurements[key]

    # step 2.4: calculate the infer time needed
    now = datetime.datetime.now()
    print('[INFO] Infer time:')
    msg = 'test at: ' + str(now) + '\n' + \
          'total infer num: ' + str(total_infer_num) + '\n' + \
          'total pure infer time(ms): ' + str(total_infer_time / 1000 / 1000) + '\n' + \
          'average pure infer time(ms): ' + str(total_infer_time / total_infer_num / 1000 / 1000) + '\n' + \
          'average pure infer time after workspace(ms): ' + str(abs(
        total_infer_time_workspace / (total_infer_num - workspace) / 1000 / 1000)) + '\n' + '\n\n\n\n'
    print(msg)

    result_txt='batch_' + str(batchsize) + '_infer_time.txt'
    with open(os.path.join(infer_res_save_path, result_txt), 'a', encoding='utf-8') as f_infer_time:
        f_infer_time.write(msg)
'''-----------------------------------------------------------------------------------'''