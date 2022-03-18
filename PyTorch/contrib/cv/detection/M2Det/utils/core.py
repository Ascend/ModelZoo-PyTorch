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
import time,os
import torch
import shutil
import argparse
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from layers.modules import MultiBoxLoss
from data import mk_anchors
from data import COCODetection, VOCDetection, detection_collate, preproc
from configs.CC import Config
from termcolor import cprint
from utils.nms_wrapper import nms
import numpy as np
#NPU修改开始
import torch.npu
#NPU修改结束

def set_logger(status):
    if status:
        from logger import Logger
        date = time.strftime("%m_%d_%H_%M") + '_log'
        log_path = './logs/'+ date
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.makedirs(log_path)
        logger = Logger(log_path)
        return logger
    else:
        pass

def anchors(cfg):
    return mk_anchors(cfg.model.input_size,
                               cfg.model.input_size,
                               cfg.model.anchor_config.size_pattern, 
                               cfg.model.anchor_config.step_pattern)
    
def init_net(net, cfg, resume_net):    
    if cfg.model.init_net and not resume_net:
        net.init_model(cfg.model.pretrained)
    else:
        print('Loading resume network...')
        state_dict = torch.load(resume_net, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict,strict=False)

def set_optimizer(net, cfg):
    return optim.SGD(net.parameters(),
                     lr = cfg.train_cfg.lr[0],
                     momentum = cfg.optimizer.momentum,
                     weight_decay = cfg.optimizer.weight_decay)

def set_criterion(cfg):
    return MultiBoxLoss(cfg.model.m2det_config.num_classes,
                        overlap_thresh = cfg.loss.overlap_thresh,
                        prior_for_matching = cfg.loss.prior_for_matching,
                        bkg_label = cfg.loss.bkg_label,
                        neg_mining = cfg.loss.neg_mining,
                        neg_pos = cfg.loss.neg_pos,
                        neg_overlap = cfg.loss.neg_overlap,
                        encode_target = cfg.loss.encode_target)#NPU更改

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size, cfg):
    global lr
    if epoch <= 5:
        lr = cfg.train_cfg.end_lr + (cfg.train_cfg.lr[0]-cfg.train_cfg.end_lr)\
         * iteration / (epoch_size * cfg.train_cfg.warmup)
    else:
        for i in range(len(cfg.train_cfg.step_lr.COCO)):
            if cfg.train_cfg.step_lr.COCO[i]>=epoch:
                lr = cfg.train_cfg.lr[i]
                break
        # lr = cfg.train_cfg.init_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_dataloader(cfg, dataset, setname='train_sets'):
    _preproc = preproc(cfg.model.input_size, cfg.model.rgb_means, cfg.model.p)
    Dataloader_function = {'VOC': VOCDetection, 'COCO':COCODetection}
    _Dataloader_function = Dataloader_function[dataset]
    if setname == 'train_sets':
        dataset = _Dataloader_function(cfg.COCOroot if dataset == 'COCO' else cfg.VOCroot,
                                   getattr(cfg.dataset, dataset)[setname], _preproc)
    else:
        dataset = _Dataloader_function(cfg.COCOroot if dataset == 'COCO' else cfg.VOCroot,
                                   getattr(cfg.dataset, dataset)[setname], None)
    return dataset
    
def print_train_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss_L:{:.4f}||Loss_C:{:.4f}||Batch_Time:{:.4f}||LR:{:.7f}'.format(*info_list), 'green')
       
def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info,str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info,list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)


def save_checkpoint(net, cfg, final=True, datasetname='COCO',epoch=10):
    if final:
        torch.save(net.state_dict(), cfg.model.weights_save + \
                'Final_M2Det_{}_size{}_net{}.pth'.format(datasetname, cfg.model.input_size, cfg.model.m2det_config.backbone))
    else:
        torch.save(net.state_dict(), cfg.model.weights_save + \
                'M2Det_{}_size{}_net{}_epoch{}.pth'.format(datasetname, cfg.model.input_size, cfg.model.m2det_config.backbone,epoch))



def write_logger(info_dict,logger,iteration,status):
    if status:
        for tag,value in info_dict.items():
            logger.scalar_summary(tag, value, iteration)
    else:
        pass

def image_forward(img, net, cuda, priors, detector, transform):
    ############## npu modify begin #############
    #CALCULATE_DEVICE = "npu:{}".format(args.gpu)
    #torch.npu.set_device(CALCULATE_DEVICE)
    ############## npu modify begin #############
    
    w,h = img.shape[1],img.shape[0]
    scale = torch.Tensor([w,h,w,h])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        x = x.npu()
        scale = scale.npu()

    out = net(x)
    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0] * scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    return boxes, scores
   
def nms_process(num_classes, i, scores, boxes, cfg, min_thresh, all_boxes, max_per_image):
    #print('nms_process开始')
    for j in range(1, num_classes): # ignore the bg(category_id=0)
        inds = np.where(scores[:,j] > min_thresh)[0]
        if len(inds) == 0:
            all_boxes[j][i] = np.empty([0,5], dtype=np.float32)
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)

        soft_nms = cfg.test_cfg.soft_nms
        keep = nms(c_dets, cfg.test_cfg.iou, force_cpu=soft_nms)
        keep = keep[:cfg.test_cfg.keep_per_class] # keep only the highest boxes
        c_dets = c_dets[keep, :]
        all_boxes[j][i] = c_dets
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]
    #print('nms_process开始')

##-自定义logr函数
def set_train_log():

     date = time.strftime("%m_%d_%H_%M") + '_loss.log'
     log_path = './logs/'+ date
     if os.path.exists(log_path):
         #os.remove(log_path)
         print('{}路径已存在'.format(log_path))
     else:
         os.mknod(log_path)
     logr = train_log(log_path)
     return logr

def train_log(filename, verbosity=1, name=None):
    import logging
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logr = logging.getLogger(name)
    logr.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logr.addHandler(fh)
    #输出到终端
    #sh = logging.StreamHandler()
    #sh.setFormatter(formatter)
    #logr.addHandler(sh)

    return logr

##-自定义计时函数
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=10):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)