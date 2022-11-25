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

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import torch
import torch.npu
import logging
import numpy as np
import time
from tqdm import tqdm
import torch.autograd.profiler as profiler
from parser1 import get_args
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from apex import amp
from utils import cos_sim
from torch.autograd import Variable
def get_logger(filename, verbosity = 1, name = None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s%(message)s]"            
            )
    logger = logging.getLogger(name)   
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger         

def train_model(model, db_gen, optimizer, epoch, args, device, lr_scheduler, criterion):    
    with open(args.save_dir + args.name + '/' + args.save_log + 'TA_{}.log'.format(epoch), mode = 'w') as ff:
        logger = get_logger(args.save_dir + args.name + '/' + args.save_log + 'TA_{}.log'.format(epoch))
    
    logger.info('Start Training!')
    
    args = get_args()
    model.train()
    count = 0
    with tqdm(total = len(db_gen), ncols = 70) as pbar:
        
        for m_batch, m_label in db_gen:

            # synchronize the process timing, neglect the first five steps
            count += 1
            if count == 6:
                torch.npu.synchronize(device)
                start = time.time()

            m_batch, m_label = m_batch.to(device, non_blocking=False), m_label.to(device, non_blocking=False)
            output = model(m_batch, m_label)
            cce_loss = criterion['cce'](output, m_label)
            loss = cce_loss
            optimizer.zero_grad()
            if args.amp_mode:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            
            if(count % 20 == 0):
              # save the .log
               logger.info('Epoch:[{}/{}] \t loss = {:.3f} \t'.format(epoch, args.epoch, loss))
            
            pbar.set_description('epoch: %d, cce:%.3f'%(epoch, cce_loss))
            print('epoch: %d, cce:%.3f'%(epoch, cce_loss))
            pbar.update(1)
            if args.do_lr_decay:
                if args.lr_decay == 'keras': lr_scheduler.step()


        # synchronzie the process timing
        torch.npu.synchronize(device)
        end = time.time()

        # calculate the total time
        time_total = end - start
        # calculate the FPS
        FPS = args.frame / (time_total * 1000 * 8)
        logger.info('FPS:{}/epoch'.format(FPS))

        

def time_augmented_evaluate_model(mode, model, db_gen, l_utt, save_dir, epoch, l_trial, args, device):
    with open(args.save_dir + args.name + '/' + args.save_log + 'TA_{}.log'.format(epoch), mode = 'w') as ff:
        logger = get_logger(args.save_dir + args.name + '/' + args.save_log + 'TA_{}.log'.format(epoch))
    
    f_log = open(args.save_dir + args.name + '/' + args.save_log + 'TA_{}.log'.format(epoch), 'a', buffering = 1)
    args = get_args()
    if mode not in ['val','eval']: raise ValueError('mode should be either "val" or "eval"')
    model.eval()
    with torch.set_grad_enabled(False):
        #1st, extract speaker embeddings.
        l_embeddings = []
        with tqdm(total = len(db_gen), ncols = 70) as pbar:
            for m_batch in db_gen:
                l_code = []
                for batch in m_batch:
                    batch = batch.to(device)
                    code = model(x = batch, is_test=True)
                    l_code.extend(code.cpu().numpy())
                l_embeddings.append(np.mean(l_code, axis=0))
                pbar.update(1)
        d_embeddings = {}
        if not len(l_utt) == len(l_embeddings):
            print(len(l_utt), len(l_embeddings))
            exit()
        for k, v in zip(l_utt, l_embeddings):
            d_embeddings[k] = v
            
        #2nd, calculate EER
        y_score = [] # score for each sample
        y = [] # label for each sample 
        f_res = open(save_dir + 'results/{}_epoch{}.txt'.format(mode, epoch), 'w')
        
        for line in l_trial:
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))


            y_score.append(cos_sim(d_embeddings[utt_a], d_embeddings[utt_b]))
            f_res.write('{score} {target}\n'.format(score=y_score[-1],target=y[-1]))
        fpr, tpr, _ = roc_curve(y, y_score, pos_label=1)

        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        logger.info('eer:{}/epoch'.format(eer))
        f_log.write('epoch:{}; eer:{}'.format(epoch, eer))
    return eer