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

from glob import glob
import json
import sys
from ECAPA_TDNN.mel2samp_tacotron2 import Mel2SampWaveglow
from ECAPA_TDNN.prepare_batch_loader import struct_meta, write_to_csv, read_from_csv, reduce_meta, build_speaker_dict, collate_function
import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from functools import partial
from tqdm import tqdm



CONFIGURATION_FILE = 'config.json'
T_THRES = 19
DATA_SET = sys.argv[1]

with open(CONFIGURATION_FILE) as f:
    data = f.read()
    json_info = json.loads(data)

    mel_config = json_info["mel_config"]
    MEL2SAMPWAVEGLOW = Mel2SampWaveglow(**mel_config)

    hp = json_info["hp"]

    global_scope = sys.modules[__name__]

    for key in hp:
        setattr(global_scope, key, hp[key])




def load_meta(dataset, keyword='vox1'):

    if keyword == 'vox1':

        wav_files_test = sorted(glob(dataset +'/vox1_test' + '/*/*/*.wav'))
        print(f'Len. wav_files_test {len(wav_files_test)}')

        test_meta = struct_meta(wav_files_test)
        write_to_csv(test_meta, 'vox1_test.csv')
    
    return  test_meta

def get_dataloader(keyword='vox1', t_thres=19, batchsize = 16, dataset = DATA_SET):
    test_meta = load_meta(dataset, keyword)
    
    test_meta_ = [meta for meta in tqdm(test_meta) if meta[2] < t_thres]
   


    test_meta = reduce_meta(test_meta_, speaker_num=REDUCED_SPEAKER_NUM_TEST)
    print(f'Meta reduced {len(test_meta_)} => {len(test_meta)}')
    
    test_speakers = build_speaker_dict(test_meta)
    
    dataset_test = DataLoader(test_meta, batch_size=batchsize,
                              shuffle=False, num_workers=2,
                              collate_fn=partial(collate_function, 
                                                 speaker_table=test_speakers,
                                                 max_mel_length=MAX_MEL_LENGTH),
                              prefetch_factor=2,
                              pin_memory=True,
                              drop_last=True)

    return dataset_test, test_speakers

if __name__ == "__main__":
    predata_path = sys.argv[2]
    prespeaker_path = sys.argv[3]
    batchsize = int(sys.argv[4])
    dataset_test, test_speakers = get_dataloader('vox1', 19, batchsize)
    if not os.path.exists(predata_path):  #判断是否存在文件夹如果不存在则创建为文件夹
       os.makedirs(predata_path)
    if not os.path.exists(prespeaker_path):  
       os.makedirs(prespeaker_path)
    i=0
    for mels, mel_length, speakers in tqdm(dataset_test):
      i=i+1
      mels = np.array(mels).astype(np.float32)
      mels.tofile(predata_path+'mels'+str(i)+".bin")
      torch.save(speakers,prespeaker_path + 'speakers'+str(i)+".pt")
      