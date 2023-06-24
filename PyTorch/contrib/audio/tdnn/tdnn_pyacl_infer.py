# Copyright 2018 NVIDIA Corporation. All Rights Reserved.
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
# ============================================================================
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
import shutil
import argparse

import acl
import numpy as np
from tqdm import tqdm
from acl_net import AclModel

DTYPE = {
    'float32': np.float32,
    'float64': np.float64,
    'int32': np.int32,
    'int64': np.int64
}

if __name__ == '__main__':
    '''
    参数说明：
        --model_path：模型路径
        --device_id：npu id
        --cpu_run：MeasureTime类的cpu_run参数，True or False
        --sync_infer：推理方式：
                      True：同步推理
                      False：异步推理
        --workspace：类似TensorRT `workspace`参数，计算平均推理时间时排除前n次推理
        --input_info_file_path：类似benchmark的bin_info文件
        --input_dtypes：模型输入的类型，用逗号分割（`DTYPE`变量）
                        e.g. 模型只有一个输入：--input_dtypes=float32
                        e.g. 模型有多个输入：--input_dtypes=float32,float32,float32（需要和bin_info文件多输入排列一致）
        --infer_res_save_path：推理结果保存目录
        --res_save_type：推理结果保存类型，bin或npy

    info文件说明：
        因为支持动态shape，相比于benchmark的info文件，需要多加一列shape信息，e.g.
        ```
        0 ./bert_bin/input_ids_0.bin (1,512)
        0 ./bert_bin/segment_ids_0.bin (1,512)
        0 ./bert_bin/input_mask_0.bin (1,512)
        1 ./bert_bin/input_ids_1.bin (1,512)
        1 ./bert_bin/segment_ids_1.bin (1,512)
        1 ./bert_bin/input_mask_1.bin (1,512)
        ```

    Using Example:
        python3.7 pyacl_infer.py \
        --model_path=./bert_base_batch_1_sim_auto.om \
        --device_id=0 \
        --cpu_run=True \
        --sync_infer=True \
        --workspace=10 \
        --input_info_file_path=./input.info \
        --input_dtypes=int64,int64,int64 \
        --infer_res_save_path=./infer_res \
        --res_save_type=bin
    '''

    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--batch_size', required=True)
    parser.add_argument('--device_id', required=True, type=int)
    parser.add_argument('--cpu_run', required=True, choices=['True', 'False'])
    parser.add_argument('--sync_infer', required=True, choices=['True', 'False'])
    parser.add_argument('--workspace', type=int, default=0)
    parser.add_argument('--input_info_file_path', required=True)
    parser.add_argument('--input_dtypes', required=True)
    parser.add_argument('--infer_res_save_path', required=True)
    parser.add_argument('--res_save_type', required=True, choices=['bin', 'npy'])
    opt = parser.parse_args()

    # 创建模型
    measurements = {}
    om_model = AclModel(device_id=opt.device_id,
                        model_path=opt.model_path,
                        sync_infer=eval(opt.sync_infer),
                        measurements=measurements,
                        key='per_infer_time_ns',
                        cpu_run=eval(opt.cpu_run))

    # 创建目录
    if os.path.exists(opt.infer_res_save_path):
        shutil.rmtree(opt.infer_res_save_path)
    os.makedirs(opt.infer_res_save_path)

    # 读取info_file
    inputs_info = {}
    with open(opt.input_info_file_path, 'rt', encoding='utf-8') as f_info:
        line = f_info.readline()
        while line:
            line = line.rstrip('\n')
            contents = line.split()
            info = {'path': contents[1], 'shape': eval(contents[2])}
            inputs_info.setdefault(contents[0], []).append(info)
            line = f_info.readline()

    # 解析输入类型
    input_dtypes = opt.input_dtypes.split(',')
    input_dtypes = list(map(lambda x: DTYPE[x], input_dtypes))

    # 读取文件推理
    total_infer_time = 0
    total_infer_time_workspace = 0
    total_infer_num = 0
    dataset = {}
    dims_infos = {}
    bs = int(opt.batch_size)
    for key, values in inputs_info.items():
        # 构造输入
        inputs = []
        dims = []
        for idx, value in enumerate(values):
            x = np.fromfile(value['path'], dtype=input_dtypes[idx]).reshape(value['shape'])
            inputs.append((key,x))
            dims.extend((bs, value['shape'][1], value['shape'][2]))
        dims_info = {'dimCount': len(dims), 'name': '', 'dims': dims}

        # (1, 1500, 23) {'dimCount': 3, 'name': '', 'dims': [1, 1500, 23]}
        length = inputs[0][1].shape[1]
        dataset[length] = dataset.get(length,[]) + inputs
        dims_infos[length] = dims_infos.get(length,dims_info)
    
    total_inputs = []
    total_keys = []
    for k in sorted(dataset.keys()):
        total_len = len(dataset[k])
        batch_input = []
        batch_key = []
        for i, (key, ipt) in enumerate(dataset[k]):
            batch_input.append(ipt)
            batch_key.append(key)
            if (i+1) % bs == 0:
                total_inputs.append(batch_input)
                total_keys.append(batch_key)
                batch_input = []
                batch_key = []
        if batch_input != []:
            total_inputs.append(batch_input)
            total_keys.append(batch_key)

    for i, b_ipt in enumerate(tqdm(total_inputs)):
        batch_input = np.squeeze(np.array(b_ipt), axis=1)
        if batch_input.shape[0] < bs:
            batch_input = np.pad(batch_input, [(0, bs-batch_input.shape[0]), (0, 0), (0, 0)], mode='constant')

        output = om_model([batch_input], dims_infos[batch_input.shape[1]])

        total_infer_num += 1

        # 保存文件
        for j, key in enumerate(total_keys[i]):
            if opt.res_save_type == 'bin':
                output[0][j].tofile(os.path.join(opt.infer_res_save_path, key + '.' + str(0) + '.bin'))
            else:
                np.save(os.path.join(opt.infer_res_save_path, key + '.' + str(0) + '.npy'), output[0][j])

        # 计算时间
        total_infer_time += measurements['per_infer_time_ns']
        if total_infer_num > opt.workspace:
            total_infer_time_workspace += measurements['per_infer_time_ns']

    # 推理时间
    print('[INFO] Infer time:')
    msg = 'total infer num: ' + str(total_infer_num) + '\n' + \
          'total pure infer time(ms): ' + str(total_infer_time / 1000 / 1000) + '\n' + \
          'average pure infer time(ms): ' + str(total_infer_time / total_infer_num / 1000 / 1000) + '\n' + \
          'average pure infer time after workspace(ms): ' + str(abs(
        total_infer_time_workspace / (total_infer_num - opt.workspace) / 1000 / 1000)) + '\n'
    print(msg)
    with open(os.path.join(opt.infer_res_save_path, 'infer_time.txt'), 'wt', encoding='utf-8') as f_infer_time:
        f_infer_time.write(msg)