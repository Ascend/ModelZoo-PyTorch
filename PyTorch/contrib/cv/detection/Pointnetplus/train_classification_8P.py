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

import os
import sys
import torch
if torch.__version__>= '1.8.1':
      import torch_npu
import numpy as np
import pdb
import datetime
import logging
import models.provider as provider
import importlib
import shutil
import argparse
import apex
import time
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp
from models.ModelNetDataLoader import ModelNetDataLoader
from apex import amp
import torch.distributed as dist


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet2_cls_ssg]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_ssg', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--npu', default=None,type=int,help='NPU id to use.')
    parser.add_argument('--world_size',default=1,type=int,  help='number of nodes for distributed training')
    parser.add_argument('--workers', type=int, default=32, help='number of data loading workers')
    parser.add_argument('--rank',default=0,type=int,help='node rank for distributed training')
    parser.add_argument('--dist_backend',default='hccl',type=str,help='distributed backend')
    parser.add_argument('--dist_url',default='env://',type=str,help='url used to set up distributed training')
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes")
    parser.add_argument("--gpus", default=8, type=int, help="number of gpus per node")
    parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
    parser.add_argument('--data',type=str, default='./data/modelnet40_normal_resampled', help='data_path')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, device, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        points, target = points.to(device), target.to(device)

        points = points.transpose(2, 1)
        pred, _ = model(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count 
def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()
    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id
    return process_device_map

def main(gpu,args):
    def log_string(str):
        logger.info(str)
        print(str)

    args.process_device_map = device_id_to_process_device_map(args.device_list)
    args.npu = args.process_device_map[gpu]
    print('npu num:', args.npu)

    CALCULATE_DEVICE = "npu:{}".format(args.npu)
    torch.npu.set_device(CALCULATE_DEVICE)
    args.batch_size = int(args.batch_size / 8)   
    print("use ", CALCULATE_DEVICE)  
    ngpus_per_node = torch.npu.device_count()
    rank = args.rank*args.gpus+gpu
    print("rank:",rank)
    dist.init_process_group(backend=args.dist_backend,  
    world_size=args.world_size, rank=rank)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir+timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    data_path = args.data

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=args.world_size, rank=rank)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=True,sampler=train_sampler)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,pin_memory=False,drop_last=True)
    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification_8P.py', str(exp_dir))
    classifier = model.get_model(num_class, normal_channel=args.use_normals)#.to(CALCULATE_DEVICE)
    criterion = model.get_loss()#.to(CALCULATE_DEVICE)
    classifier.apply(inplace_relu)
    if not args.use_cpu:
        classifier = classifier.to(CALCULATE_DEVICE)
        criterion = criterion.to(CALCULATE_DEVICE)
    log_string('No existing model, starting training from scratch...')
    start_epoch = 0
    '''MODEL LOADING'''
    if args.optimizer == 'Adam':
        optimizer = apex.optimizers.NpuFusedAdam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = apex.optimiziers.NpuFusedSGD(classifier.parameters(), lr=0.01, momentum=0.9)
    classifier, optimizer = amp.initialize(classifier, optimizer, opt_level="O2", loss_scale = "dynamic", combine_grad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.npu], broadcast_buffers=False)

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        train_sampler.set_epoch(epoch)
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        error_cnt = 0
        mean_correct = []
        classifier.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        tot_time =  AverageMeter()
        end = time.time()
        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            print('===================>4')
            optimizer.zero_grad()
            data_time.update(time.time() - end)
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.to(CALCULATE_DEVICE), target.to(CALCULATE_DEVICE)

    
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            error_cnt += loss.item()
            pred_choice = pred.data.max(1)[1]
        
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            with amp.scale_loss(loss,optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            global_step += 1
            current_batch_time = time.time() - end
            batch_time.update(current_batch_time)
            end = time.time()
            FPS = (args.batch_size / current_batch_time) * 8
            
            if batch_id > 4:
                log_string("Epoch %d step %d FPS: %f" % (epoch, batch_id, FPS))
                tot_time.update(current_batch_time)
     
        epoch_FPS = args.batch_size / tot_time.avg
        if gpu == 0:
            log_string("Epoch %d avg FPS: %f" % (epoch, epoch_FPS))
            log_string("Epoch %d train loss: %f " % (epoch, error_cnt / batch_id))
            train_instance_acc = np.mean(mean_correct)
            log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier, testDataLoader, CALCULATE_DEVICE, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Acc@1: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
    logger.info('End of training...')

def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()
    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id
    return process_device_map

if __name__ == '__main__':
    arges = parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1' # 可以使用当前真实ip或者'127.0.0.1'
    os.environ['MASTER_PORT'] = '29688' # 随意一个可使用的port即可
    arges.process_device_map = device_id_to_process_device_map(arges.device_list)
    ngpus_per_node = len(arges.process_device_map)
    arges.world_size = ngpus_per_node * arges.world_size

    mp.spawn(main,args=(arges,),nprocs=arges.gpus)
