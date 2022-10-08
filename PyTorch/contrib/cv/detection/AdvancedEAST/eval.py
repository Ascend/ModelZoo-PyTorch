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

import torch
if torch.__version__>= '1.8':
    import torch_npu
import torch.utils.data
import multiprocessing
import numpy as np

from model_VGG import advancedEAST
from losses import quad_loss
from dataset import RawDataset, data_collate
from utils import Averager, eval_pre_rec_f1
import cfg

device = torch.device(cfg.device)


def eval_func(i, out, gt_xy_list):
    eval_p_r_f = eval_pre_rec_f1()
    eval_p_r_f.add(out, gt_xy_list)
    mPre, mRec, mF1_score = eval_p_r_f.val()
    np.save('val_temp/{}.npy'.format(str(i)), [mPre, mRec, mF1_score])
    eval_p_r_f.reset()


def eval():
    """ dataset preparation """
    val_dataset = RawDataset(is_val=True)

    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=data_collate,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    model = advancedEAST()
    state_dict = {k.replace('module.', ''): v for k, v in torch.load(cfg.pth_path, map_location='cpu').items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    loss_func = quad_loss
    val_loss_avg = Averager()
    val_Loss_list = []
    thread_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    i = 0
    for image_tensors, labels, gt_xy_list in valid_loader:
        batch_x = image_tensors.float().to(device)
        batch_y = labels.float().to(device)

        out = model(batch_x)
        loss = loss_func(batch_y, out)

        val_loss_avg.add(loss)
        val_Loss_list.append(val_loss_avg.val())
        thread_pool.apply_async(eval_func, args=(i, out.cpu().detach(), gt_xy_list))
        i += 1

    thread_pool.close()
    thread_pool.join()

    print('loss:{:.3f}'.format(val_loss_avg.val().item()))
    val_loss_avg.reset()

    mPre = mRec = mF1_score = 0
    size = len(valid_loader)
    for i in range(size):
        arr = np.load('val_temp/{}.npy'.format(str(i)))
        mPre += arr[0]
        mRec += arr[1]
        mF1_score += arr[2]
    mPre /= size
    mRec /= size
    mF1_score /= size
    print('precision:{:.2f}% recall:{:.2f}% f1-score:{:.2f}%'.format(mPre, mRec, mF1_score))


if __name__ == '__main__':
    os.makedirs('val_temp', exist_ok=True)
    eval()
