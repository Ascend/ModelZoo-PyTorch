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

print("+-" * 50)
import torch, os, datetime

if torch.__version__ >= "1.8":
    import torch_npu
import numpy as np

from model.model import parsingNet
from data.dataloader import get_train_loader

from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time
from apex import amp


def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.npu(), cls_label.long().npu(), seg_label.long().npu()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out': seg_out, 'seg_label': seg_label}
    else:
        img, cls_label = data_label
        img, cls_label = img.npu(), cls_label.long().npu()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results, logger, global_step):
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)
        loss_cur = loss_cur.npu()

        if global_step % 20 == 0:
            logger.add_scalar('loss/' + loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss


def train_p(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, use_aux, prof_flag):
    net.train()
    progress_bar = dist_tqdm(train_loader)
    count = 0
    a = 0
    t_data_0 = time.time()

    if prof_flag:  # 如果需要跑prof文件的话
        for b_idx, data_label in enumerate(progress_bar):
            reset_metrics(metric_dict)
            global_step = epoch * len(data_loader) + b_idx
    
            with torch.autograd.profiler.profile(use_npu=True) as prof:
                results = inference(net, data_label, use_aux)
                loss = calc_loss(loss_dict, results, logger, global_step)
                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
    
            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            prof.export_chrome_trace("/home/UFLD/output.prof")  # "output.prof"为输出文件地址
    
    
            scheduler.step(global_step)
            results = resolve_val_data(results, use_aux)
            update_metrics(metric_dict, results)
            t = time.time()
            s = 128 / (t - t_data_0)
            count += 1
            if count > 5 and epoch > 0:
                a += s
            if global_step % 20 == 0:
                for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                    logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
    
            if hasattr(progress_bar, 'set_postfix'):
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
                progress_bar.set_postfix(loss='%.3f' % float(loss), **kwargs)
            t_data_0 = time.time()
        if epoch > 0:
            print("Epoch: " + str(epoch) + " FPS = " + str(a / (count - 5)))
    else:
        for b_idx, data_label in enumerate(progress_bar):
            reset_metrics(metric_dict)
            global_step = epoch * len(data_loader) + b_idx

            results = inference(net, data_label, use_aux)
            loss = calc_loss(loss_dict, results, logger, global_step)
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            scheduler.step(global_step)
            results = resolve_val_data(results, use_aux)
            update_metrics(metric_dict, results)
            t = time.time()
            s = 128 / (t - t_data_0)
            count += 1
            if count > 4 and epoch > 0:
                a += s
            if global_step % 20 == 0:
                for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                    logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            if hasattr(progress_bar, 'set_postfix'):
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
                progress_bar.set_postfix(loss='%.3f' % float(loss), **kwargs)
            t_data_0 = time.time()
        if epoch > 0:
            print("Epoch: " + str(epoch) + " FPS = " + str(a / (count - 4)))


def train_8p(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, use_aux):
    net.train()
    progress_bar = dist_tqdm(train_loader)
    count = 0
    a = 0
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
        # t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx
        # t_net_0 = time.time()
        results = inference(net, data_label, use_aux)
        loss = calc_loss(loss_dict, results, logger, global_step)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        # t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)
        update_metrics(metric_dict, results)
        t = time.time()
        s = 128 * 8 / (t - t_data_0)
        count += 1
        if count > 1 and epoch > 0:
            a += s
        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
        if args.local_rank == 0:
            if hasattr(progress_bar, 'set_postfix'):
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in
                          zip(metric_dict['name'], metric_dict['op'])}
                progress_bar.set_postfix(loss='%.3f' % float(loss),
                                         # data_time='%.3f' % float(t_data_1 - t_data_0),
                                         # net_time='%.3f' % float(t_net_1 - t_net_0),
                                         **kwargs)
        t_data_0 = time.time()
    if args.local_rank == 0 and epoch > 0:
        print("Epoch: " + str(epoch) + " FPS: " + str(a / (count - 1)))


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.npu.set_device(args.local_rank)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29680'
        torch.distributed.init_process_group(backend='hccl', world_size=int(os.environ['WORLD_SIZE']), rank=args.local_rank)
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    train_loader, cls_num_per_lane, sample = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num,
                                                              cfg.dataset,
                                                              cfg.use_aux, distributed, cfg.num_lanes)

    net = parsingNet(pretrained=True, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes), use_aux=cfg.use_aux).npu()
    optimizer = get_optimizer(net, cfg)
    model, optimizer = amp.initialize(net, optimizer, opt_level="O1", loss_scale="dynamic", combine_grad=True)
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], broadcast_buffers=False)

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k, v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    cp_projects(args.auto_backup, work_dir)
    for epoch in range(resume_epoch, cfg.epoch):
        if distributed:
            sample.set_epoch(epoch)
            train_8p(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg.use_aux)
        else:
            train_p(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg.use_aux, False)
        save_model(net, optimizer, epoch, work_dir, distributed)
    logger.close()
