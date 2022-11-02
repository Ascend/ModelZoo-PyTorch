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
import models
import datas
import argparse
import torch
import torchvision.transforms as TF
import torch.nn as nn
import os
import sys
import time
from utils.config import Config


print(f"initial network, it might take minutes.")
CALCULATE_DEVICE = "npu: 0"
torch.npu.set_device(CALCULATE_DEVICE)
# loading configures
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="configs/test_config")
parser.add_argument("--data_url", type=str, default="/home/ma-user/modelarts/inputs/data_url_0")
args = parser.parse_args()
config = Config.from_file(args.config)
# preparing datasets
normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
normalize2 = TF.Normalize([0, 0, 0], config.std)
trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

revmean = [-x for x in config.mean]
revstd = [1.0 / x for x in config.std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])
revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])
testset = datas.AIMSequence(config.testset_root, trans, config.test_size, config.test_crop_size, config.inter_frames)
sampler = torch.utils.data.SequentialSampler(testset)
validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=0)


# model
model = getattr(models, config.model)(config.pwc_path).npu()
model = nn.DataParallel(model)
tot_time = 0
tot_frames = 0
print('Everything prepared. Ready to test...')
to_img = TF.ToPILImage()


def generate():
    global tot_time, tot_frames
    store_path = config.store_path
    with torch.no_grad():
        for validationIndex, validationData in enumerate(validationloader, 0):
            print('Testing {}/{}-th group...'.format(validationIndex, len(testset)))
            sys.stdout.flush()
            sample, folder, index = validationData

            # make sure store path exists
            if not os.path.exists(config.store_path + '/' + folder[1][0]):
                os.mkdir(config.store_path + '/' + folder[1][0])

            # if sample consists of four frames (ac-aware)
            if len(sample) == 4:
                frame0 = sample[0]
                frame1 = sample[1]
                frame2 = sample[-2]
                frame3 = sample[-1]

                I0 = frame0.npu()
                I3 = frame3.npu()

                I1 = frame1.npu()
                I2 = frame2.npu()

                revtrans(I1.clone().cpu()[0]).save(store_path + '/' + folder[1][0] + '/'  + index[1][0] + '.png')
                revtrans(I2.clone().cpu()[0]).save(store_path + '/' + folder[-2][0] + '/' +  index[-2][0] + '.png')
            # else two frames (linear)
            else:
                frame0 = None
                frame1 = sample[0]
                frame2 = sample[-1]
                frame3 = None

                I0 = None
                I3 = None
                I1 = frame1.npu()
                I2 = frame2.npu()
             
                revtrans(I1.clone().cpu()[0]).save(store_path + '/' + folder[0][0] + '/'  + index[0][0] + '.png')
                revtrans(I2.clone().cpu()[0]).save(store_path + '/' + folder[1][0] + '/' +  index[1][0] + '.png')

            
            for tt in range(config.inter_frames):
                x = config.inter_frames
                t = 1.0/(x+1) * (tt + 1)
                # print(t)


                # record duration time
                start_time = time.time()

                output = model(I0, I1, I2, I3, t)
                It_warp = output
                
                tot_time += (time.time() - start_time)
                tot_frames += 1
                

                if len(sample) == 4:
                    revtrans(It_warp.cpu()[0]).save(store_path + '/' + folder[0][0] + '/' + index[1][0] + '_' + str(tt) + '.png')
                else:
                    revtrans(It_warp.cpu()[0]).save(store_path + '/' + folder[0][0] + '/' + index[0][0] + '_' + str(tt) + '.png')


def test():
    print(f"loading {config.checkpoint} to {CALCULATE_DEVICE}.")
    dict1 = torch.load(config.checkpoint, map_location=CALCULATE_DEVICE)
    model.load_state_dict(dict1['model_state_dict'])
    if not os.path.exists(config.store_path):
        os.mkdir(config.store_path)
    generate()


test()
print('Avg time is {} second'.format(tot_time/tot_frames))
