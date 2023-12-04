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

import os
import argparse
import numpy as np
import torch
import torch_aie
from torch_aie import _enums
from glob import glob
from ECAPA_TDNN.prepare_batch_loader import struct_meta, write_to_csv, read_from_csv, reduce_meta, build_speaker_dict, collate_function
from torch.utils.data import DataLoader
from functools import partial
from model_pt import forward_nms_script


def load_meta(dataset, keyword='vox1'):
    if keyword == 'vox1':
        wav_files_test = sorted(glob(dataset + '/*/*/*.wav'))
        print(f'Len. wav_files_test {len(wav_files_test)}')
        test_meta = struct_meta(wav_files_test)
    return  test_meta

def get_dataloader(keyword='vox1', t_thres=19, batchsize = 16, dataset = "VoxCeleb1"):
    test_meta = load_meta(dataset, keyword)
    test_meta_ = [meta for meta in (test_meta) if meta[2] < t_thres]
    test_meta = reduce_meta(test_meta_, speaker_num=-1)
    print(f'Meta reduced {len(test_meta_)} => {len(test_meta)}')
    test_speakers = build_speaker_dict(test_meta)
    dataset_test = DataLoader(test_meta, batch_size=batchsize,
                              shuffle=False, num_workers=1,
                              collate_fn=partial(collate_function,
                                                 speaker_table=test_speakers,
                                                 max_mel_length=200),
                              drop_last=True)
    return dataset_test, test_speakers

def main(opt):
    # load model
    model = torch.jit.load(opt.model)
    torch_aie.set_device(opt.device_id)
    if opt.need_compile:
        inputs = []
        inputs.append(torch_aie.Input((opt.batch_size, 80, 200)))
        model = torch_aie.compile(
            model,
            inputs=inputs,
            precision_policy=_enums.PrecisionPolicy.FP16,
            truncate_long_and_double=True,
            require_full_compilation=False,
            allow_tensor_replace_int=False,
            min_block_size=3,
            torch_executed_ops=[],
            soc_version=opt.soc_version,
            optimization_level=0)

    # load dataset
    dataloader, test_speakers = get_dataloader('vox1', 19, opt.batch_size, dataset=opt.data_path)
    # inference & nms
    pred_results = forward_nms_script(model, dataloader, opt.batch_size, opt.device_id)
    output_folder = f"result/output_bs{opt.batch_size}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for index, res in enumerate(pred_results):
        for i, r in enumerate(res):
            result_fname = 'mels' + str(index * opt.batch_size + i + 1) + '_0.bin'
            np.array(r.numpy().tofile(os.path.join(output_folder, result_fname)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv3 offline model inference.')
    parser.add_argument('--data_path', type=str, default="VoxCeleb1", help='root dir for val images and annotations')
    parser.add_argument('--soc_version', type=str, default='Ascend310P3', help='soc version')
    parser.add_argument('--model', type=str, default="ecapa_tdnn_torch_aie_bs1.pt", help='ts model path')
    parser.add_argument('--need_compile', action="store_true", help='if the loaded model needs to be compiled or not')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    opt = parser.parse_args()
    main(opt)
