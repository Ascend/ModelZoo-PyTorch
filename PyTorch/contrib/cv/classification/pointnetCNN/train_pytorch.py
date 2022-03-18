# Copyright 2021 Huawei Technologies Co., Ltd
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
import argparse
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import logging
import provider
import math
import random
import time

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


from utils.model import RandPointCNN
from utils.util_funcs import knn_indices_func_gpu, knn_indices_func_cpu
from utils.util_layers import Dense
from apex import amp

random.seed(0)
dtype = torch.cuda.FloatTensor

rank = -1

world_size = 8
dist_backend = 'nccl'

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,default='data/modelnet40_ply_hdf5_2048/')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--multi',type=int,default=1)
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.008, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--local_rank', type=int, default=1, help='local_Rank')
FLAGS = parser.parse_args()

NUM_EPOCHS = FLAGS.epoch
amp_mode = True

rank = FLAGS.local_rank

torch.npu.set_device(rank)

use8p = FLAGS.multi

NUM_POINT = FLAGS.num_point
LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
       
MAX_NUM_POINT = 2048

DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

data_path = FLAGS.data_path

LEARNING_RATE_MIN = 0.00001
        
NUM_CLASS = 40
BATCH_SIZE = FLAGS.batch_size #32

jitter = 0.01
jitter_val = 0.01

rotation_range = [0, math.pi / 18, 0, 'g']
rotation_rage_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']


world_size = 8
dist_backend = 'hccl'

class modelnet40_dataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]


AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, NUM_CLASS, with_bn=False, activation=None)
        )

    def forward(self, x):
        x = self.pcnn1(x)
        if False:
            print("Making graph...")
            k = make_dot(x[1])

            print("Viewing...")
            k.view()
            print("DONE")

            assert False
        x = self.pcnn2(x)[1]  # grab features
        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean


print("------Building model-------")
model = Classifier().npu()
print("------Successfully Built model-------")


if use8p:
    torch.distributed.init_process_group(backend=dist_backend, world_size=world_size, rank=rank)


optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = 0.9)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",loss_scale='dynamic')
for ls in amp._amp_state.loss_scalers:
    ls._scale_seq_len=20
if use8p:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)

loss_fn = nn.CrossEntropyLoss()

global_step = 1

TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, data_path,'train_files.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, data_path,'test_files.txt'))

losses = []
accuracies = []

loss_data = []
loss_av = 0


acc = 0
total = 0
def train():
    if use8p:
        LEARNING_RATE=0.008
    else:
        LEARNING_RATE=0.001
    global_step = 1
    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch > 1:
            loss_av = sum_av / step_num
            loss_data.append(loss_av)
        train_file_idxs = np.arange(0, len(TRAIN_FILES))
        np.random.shuffle(train_file_idxs)
        step_num = 0
        sum_av = 0
        loss_avg = 0
        steps = 0
        for fn in range(len(TRAIN_FILES)):
            current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
            current_data = current_data[:, 0:NUM_POINT, :]

            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
            current_label = np.squeeze(current_label)

            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE

            if epoch > 1:
                if LEARNING_RATE > LEARNING_RATE_MIN:
                    LEARNING_RATE *= DECAY_RATE ** (global_step // DECAY_STEP)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = LEARNING_RATE
                if epoch % 20 == 0:
                    torch.save(model.state_dict(), 'pointcnn_epoch{}.pth'.format(epoch))
            if use8p:
                numb_start = int(num_batches // 8 * rank)
                numb_end = int((num_batches // 8) * (rank + 1))
            else:
                numb_start = 0
                numb_end = num_batches
            for batch_idx in range(numb_start, numb_end):
                step_num += 1
                if step_num > 2:
                    epoch_start_time = time.time()
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE
                label = current_label[start_idx:end_idx]
                label = torch.from_numpy(label).long()
                label = Variable(label, requires_grad=False).npu()
                rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
                jittered_data = provider.jitter_point_cloud(rotated_data)
                P_sampled = jittered_data
                optimizer.zero_grad()

                t0 = time.time()
                P_sampled = torch.from_numpy(P_sampled).float()
                P_sampled = Variable(P_sampled, requires_grad=False).npu()
                out = model((P_sampled, P_sampled))
                loss = loss_fn(out, label)
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                #print("epoch:" + str(epoch) + "   loss: " + str(loss.item()))
                
                #print("EPOCH:{}STEP:{}LOSS: {} FPS:{:.3f}".format(epoch, step_num, loss_avg / steps,8 * BATCH_SIZE / (time.time() - epoch_start_time)))
                if global_step % 25 == 0:
                    loss_v = loss.item()
                else:
                    loss_v = 0
                loss_avg += loss.item()
                steps += 1
                if steps == 37 and use8p:
                    #logging.basicConfig(filename='910A_8p.log', level=logging.INFO)
                    print("EPOCH: {} STEP: {} LOSS: {} FPS:{:.3f}".format(epoch, step_num, loss_avg / steps,
                                                                                 8 * BATCH_SIZE / (
                                                                                         time.time() - epoch_start_time)))
                if use8p == False and step_num > 2:
                    #logging.basicConfig(filename='910A_1p.log', level=logging.INFO)
                    print("EPOCH: {} STEP: {} LOSS: {} FPS:{:.3f}".format(epoch, step_num, loss.item(),
                                                                                 BATCH_SIZE / (
                                                                                         time.time() - epoch_start_time)))
                global_step += 1
                sum_av += loss.item()
if __name__ == '__main__':
    train()

