# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import apex
import argparse 
import logging
import os 
import sys 
import time 
import torch 
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn 
import torch.nn.parallel as par
import torch.optim as optim

sys.path.append('.')
import checkpoint
import data
import lsr
import utils 
import vovnet


def main(args):
    main_worker(args)


def main_worker(args):
    if args.local_rank == 0:
        init_logger(args.log_file)

    if args.distributed:
        args.device = '{}:{}'.format(args.device, args.local_rank)
        dist.init_process_group(
            backend='hccl' if 'npu' in args.device else 'nccl',
            world_size=args.num_devices,
            rank=args.local_rank
        )

    if 'cuda' in args.device:
        cudnn.benchmark = True 
        torch.cuda.set_device(args.device)
    elif 'npu' in args.device:
        torch.npu.set_device(args.device)

    if args.eval_from_local:
        eval_from_local(args)
        exit(0)

    model = create_model(args.net, args.num_classes)

    if args.label_smoothing_epsilon > 0.0:
        logging.info('Using label smoothing with epsilon = {:.3f}'.format(args.label_smoothing_epsilon))
        criterion = lsr.get_label_smoothing_cross_entropy(
            args.device, 
            num_classes=1000,
            smooth_factor=args.label_smoothing_epsilon
        )
    else:
        logging.info('Label smoothing is unused')
        criterion = nn.CrossEntropyLoss()

    optimizer_kwargs = {
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay
    }
    if 'npu' in args.device:
        from apex.optimizers import NpuFusedSGD
        logging.info('Using NpuFusedSGD as optimizer. args = {}'.format(str(optimizer_kwargs)))
        optimizer = NpuFusedSGD(model.parameters(), **optimizer_kwargs)
    else:
        logging.info('Using optim.SGD as optimizer, args = {}'.format(str(optimizer_kwargs)))
        optimizer = optim.SGD(model.parameters(), **optimizer_kwargs)

    scheduler = get_lr_scheduler(optimizer, args)

    model.to(args.device)
    criterion.to(args.device)

    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')
    train_sampler, train_loader = data.create_train_loader(
        train_dir, args, distributed=args.distributed
    )
    scale = 0.875 # 224 / 256
    _, val_loader = data.create_val_loader(
        val_dir, args, scale, distributed=args.distributed
    )

    model, optimizer = apex.amp.initialize(
        model, optimizer, 
        opt_level=args.opt_level, 
        loss_scale=args.loss_scale
    )
    if args.distributed:
        model = par.DistributedDataParallel(model, device_ids=[args.local_rank])

    if args.start_epoch > 1:
        with utils.BlockTimer(
                args.device,
                'loading from checkpoint after epoch #{}'.format(args.start_epoch - 1)
                ):
            checkpoint.load_model(model, optimizer, args.start_epoch - 1, device=args.device)

    if args.fine_tune_from is not None:
        state_dict = utils.load_state_dict(args.fine_tune_from, map_location=args.device)
        if state_dict['last_linear.bias'].shape[0] != args.num_classes:
            modify_num_classes(state_dict, args.num_classes, args.device)
        model.load_state_dict(state_dict)

    if args.profile:
        profile(train_loader, model, criterion, optimizer, args.device)
        exit(0)

    # Training process
    print("{} starts training.".format(args.device))
    for epoch_id in range(max(1, args.start_epoch), args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch_id)
        # log current learning rate
        cur_lr = optimizer.param_groups[0]['lr']
        logging.info('Learning rate of epoch #{} is {:.5f}'.format(epoch_id, cur_lr))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch_id, args)
        # save checkpoint 
        if args.local_rank == 0 and epoch_id % args.save_freq == 0:
            with utils.BlockTimer(args.device, 'saving checkpoint after epoch #{}'.format(epoch_id)):
                checkpoint.save_checkpoint(model, optimizer, epoch_id)
        # evaluate on validation set
        validate(val_loader, model, criterion, args, epoch_id=epoch_id)
        # schedule learning rate
        scheduler.step()
    
    # save final training result
    if args.local_rank == 0:
        utils.save_state_dict(model.state_dict(), args.model)


def create_model(name:str, num_classes=1000, use_pretrained=False):
    if name.lower() == 'vovnet-39':
        print("Creating model 'VoVNet-39'")
        model = vovnet.vovnet_39(num_classes=num_classes)
    else:
        raise NotImplementedError(f"Model '{name}' is not implemented yet!")

    return model 


def train(train_loader, model, criterion, optimizer, epoch_id, args):
    batch_time, losses, top1, top5, fps = utils.make_average_meters(5)
    # switch to train mode 
    model.train()

    last_time = time.time()
    for i, (input, target) in enumerate(train_loader, start=1):
        if 'npu' in args.device:
            target = target.to(torch.int32)                      
        input = input.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        # compute output, loss and validation accuracy
        output = model(input)
        loss = criterion(output, target)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5), num_devices=args.num_devices)
        # update average loss and validation accuracy
        if args.distributed:
            # reduce for distributed training (take average of devices)
            reduced_loss = utils.reduce_tensor(loss.data, args.num_devices)
            losses.update(reduced_loss.data.item(), input.size(0))
        else:
            losses.update(loss.data.item(), input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        # measure elapsed time (ignore first 2 steps)
        if i >= 2:
            batch_time.update(time.time() - last_time)
            fps.update(args.num_devices * input.size(0) / (time.time() - last_time))
        last_time = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            log_info = 'E{}-B{}/{}\t\t'.format(epoch_id, i, len(train_loader)) + \
                       'Time: {:.3f}\t'.format(batch_time.val) + \
                       'FPS: {:.3f}\t'.format(fps.val) + \
                       'Loss: {:.4f}\t'.format(losses.val) + \
                       'Acc@1: {:.1f}\t'.format(top1.val) + \
                       'Acc@5: {:.1f}'.format(top5.val)
            logging.info(log_info)

        if args.debug and i > 5 * args.print_freq:
            break

    if args.local_rank == 0:
        log_info = 'Training Epoch #{}: \t'.format(epoch_id) + \
                'Avg. Batch time: {:.3f}\t'.format(batch_time.avg) + \
                'Avg. FPS: {:.3f}\t'.format(fps.avg) + \
                'Avg. Loss: {:.3f}'.format(losses.avg)
        logging.info(log_info)


def validate(val_loader, model, criterion, args, epoch_id=None):
    with torch.no_grad():
        losses, top1, top5, batch_time, fps = utils.make_average_meters(5)
        # switch to evaluate mode
        model.eval()

        last_time = time.time()
        for i, (input, target) in enumerate(val_loader, start=1):
            if 'npu' in args.device:     
                target = target.to(torch.int32)                      
            input = input.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            # compute output
            output = model(input)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5), num_devices=args.num_devices)
            # reduce for distributed training
            if args.distributed:
                loss = utils.reduce_tensor(loss.data, args.num_devices)
            # measure accuracy and record loss 
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time (ignore first 2 steps)
            if i >= 2:
                batch_time.update(time.time() - last_time)
                fps.update(args.num_devices * input.size(0) / (time.time() - last_time))
            last_time = time.time()

            if args.debug and i > 5 * args.print_freq:
                break

        if args.local_rank == 0:
            logging.info('--------')
            log_info = 'Validation{}: \t'.format(
                            '' if epoch_id is None else ' Epoch #{}'.format(epoch_id)
                        ) + \
                    'Avg. Acc@1: {:.3f}\t'.format(top1.avg) + \
                    'Avg. Acc@5: {:.3f}\t'.format(top5.avg) + \
                    'Avg. Eval. FPS: {:.3f}'.format(fps.avg)
            logging.info(log_info)
            logging.info('--------')

        return top1.avg, top5.avg


def profile(train_loader, model, criterion, optimizer, device):
    # switch to train mode 
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        input_var = autograd.Variable(input)
        target_var = autograd.Variable(target)

        def do_iteration():
            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer.zero_grad()
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        
        # compute gradient and do SGD step
        if i >= 5:
            kwargs = {'use_npu' if 'npu' in device else 'use_cuda': True}
            with torch.autograd.profiler.profile(**kwargs) as prof:
                do_iteration()
            prof.export_chrome_trace('output.prof')
            return
        else:
            do_iteration()


def eval_from_local(args):
    model = create_model(args.net, num_classes=args.num_classes)
    state_dict = utils.load_state_dict(args.eval_model_path)
    model.load_state_dict(state_dict)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    val_loader = get_eval_val_loader(model, args)

    if args.distributed:
        model = par.DistributedDataParallel(model, device_ids=[args.local_rank])

    validate(val_loader, model, criterion, args)


def get_eval_val_loader(model, args):
    val_dir = os.path.join(args.data, 'val')
    scale = 0.875 # 224 / 256
    _, val_loader = data.create_val_loader(
        val_dir, args, scale, distributed=args.distributed
    )
    return val_loader


def accuracy(output, target, topk=(1), num_devices=1):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred_indices = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred_indices = pred_indices.t()
    correct = pred_indices.eq(target.view(1, -1).expand_as(pred_indices))

    res = []
    for k in topk:
        st = correct[0:k].float().sum()
        if num_devices > 1:
            st = utils.reduce_tensor(st.data, num_devices)
        topk_acc = st.item() / batch_size * 100.0 # in percentage
        res.append(topk_acc)
    return res


def get_lr_scheduler(optimizer, args):
    if args.scheduler == 'step':
        logging.info('Using optim.lr_scheduler.StepLR with step = {}, gamma = {:.3f}'.format(
            args.lr_step_n, 
            args.lr_step_gamma
        ))
        return optim.lr_scheduler.StepLR(
            optimizer, 
            args.lr_step_n, 
            gamma=args.lr_step_gamma, 
            last_epoch=args.start_epoch
        )
    elif args.scheduler == 'cosine':
        logging.info('Using optim.lr_scheduler.CosineAnnealingLR with T_max = {}, eta_min = {:.3f}'.format(
            args.lr_cosine_T_max,
            args.lr_cosine_eta_min
        ))
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.lr_cosine_T_max,
            eta_min=args.lr_cosine_eta_min,
            last_epoch=args.start_epoch
        )
    else:
        raise NotImplementedError('Scheduler \'{}\' not implemented!'.format(args.scheduler))


def modify_num_classes(state_dict, to_num_classes, to_device):
    weight = state_dict['last_linear.weight']
    bias = state_dict['last_linear.bias']
    from_num_classes = bias.shape[0]
    diff = abs(from_num_classes - to_num_classes)

    if from_num_classes < to_num_classes:
        weight_added = torch.zeros([diff, weight.shape[1]]).to(to_device)
        bias_added = torch.zeros(diff).to(to_device)
        state_dict['last_linear.weight'] = torch.cat([weight, weight_added], dim=0)
        state_dict['last_linear.bias'] = torch.cat([bias, bias_added], dim=0)
    elif from_num_classes > to_num_classes:
        state_dict['last_linear.weight'] = weight[0:to_num_classes, :]
        state_dict['last_linear.bias'] = weight[0:to_num_classes]


def init_logger(filename=None):
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m-%d %I:%M:%S')

    if filename is None:
        if not os.path.exists('./log'):
            os.mkdir('./log')
        t = time.time()
        local_time = time.localtime(t)
        filename = os.path.join('log/train-{}{t.tm_mon:02}{t.tm_mday:02}' \
                                '{t.tm_hour:02}{t.tm_min:02}{t.tm_sec:02}'.format(
            local_time.tm_year % 100, t=local_time
        ))
    
    fh = logging.FileHandler(filename)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('Logger starts')


def parse_args():
    parser = argparse.ArgumentParser(description="Huawei Ascend Neural Networks in PyTorch")

    parser.add_argument('--net', type=str, metavar='NAME', 
                        choices=['senet-154', 'vovnet-39'],
                        help='model type')

    parser.add_argument('--fine-tune-from', type=str, default=None, metavar='PATH',
                        help='Model to fine-tune from (default: None)')

    parser.add_argument('--data', default="./data", type=str, metavar='DIR',
                        help='path to dataset')

    parser.add_argument('--model', default='./model.pth', type=str, metavar='PATH',
                        help='save path to final model')

    parser.add_argument('--log-file', type=str, default=None, metavar='NAME',
                        help='log file name (default: None)')

    parser.add_argument('--device', type=str, default='cpu',
                        help="device used for 1P training ('cpu', 'cuda:0', 'npu:0', etc.), " \
                             "or device type ('cuda' or 'npu') for 8P training (default: 'cpu')")

    parser.add_argument('--distributed', dest='distributed', action='store_true', default=False,
                        help='enables distributed mode')

    parser.add_argument('--num-devices', type=int, default=8,
                        help='number of devices in distributed mode (default: 8)')

    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', default=0, type=int, metavar='N',
                        help='local rank of this process (default: 0, used for torch.distributed.launch)')

    parser.add_argument('--num-workers', default=8, type=int,
                        help='number of workers for loading data (default: 8)')

    parser.add_argument('--opt-level', default='O2', type=str, choices=['O1', 'O2'],
                        help='optimization level of amp.initialize (default: \'O2\')')

    parser.add_argument('--loss-scale', default=128, type=int,
                        help='loss scale of amp.initialize (default: 128)')

    parser.add_argument('--num-classes', default=1000, type=int, 
                        help='number of classification classes (default: 1000)')

    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')

    parser.add_argument('--start-epoch', default=-1, type=int, metavar='N',
                        help='manual epoch number, checkpoint will be load if start epoch > 1 (default: -1)')

    parser.add_argument('--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')

    parser.add_argument('--lr', '--initial-lr', '--learning-rate', dest='lr', default=0.6, type=float,
                        metavar='LR', help='initial learning rate (default: 0.6)')

    parser.add_argument('--scheduler', '--lr-scheduler', default='step', type=str, choices=['step', 'cosine'],
                        help='type of scheduler (default: \'step\')')

    parser.add_argument('--lr-step-n', default=30, type=int,
                        help='number of epochs per step in StepLR scheduler (default: 30)')

    parser.add_argument('--lr-step-gamma', default=0.1, type=float,
                        help='learning rate decay ratio in StepLR scheduler (default: 0.1)')

    parser.add_argument('--lr-cosine-t-max', dest='lr_cosine_T_max', default=None, type=int,
                        help='T_max of CosineAnnealingLR scheduler (default: =epochs)')

    parser.add_argument('--lr-cosine-eta-min', default=0.0, type=float,
                        help='eta_min of CosineAnnealingLR scheduler (default: 0.0)')

    parser.add_argument('--label-smoothing-epsilon', '--label-smoothing-factor', default=0.0, type=float,
                        help='epsilon for smooth labeling (default: 0.0)')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')

    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--save-freq', default=1, type=int, metavar='N',
                        help='checkpoint saving frequency (default: 1)')

    parser.add_argument('--eval-from-local', '--evaluate-from-local',  dest='eval_from_local', default=False,
                        action='store_true', help='evaluate local model on validation set (default: False)')

    parser.add_argument('--eval-model-path', '--evaluate-model-path', dest='eval_model_path', 
                        default=None, type=str, metavar='PATH',
                        help='Path of local model to be evaluated (default: None)')

    parser.add_argument('--profile', dest='profile', default=False,
                        action='store_true', help='run in profile mode (default: False)')

    parser.add_argument('--debug', dest='debug', default=False,
                        action='store_true', help='debug mode for checking correctness of code (default: False)')

    parser.add_argument('--do-not-preserve-aspect-ratio',
                        dest='preserve_aspect_ratio',
                        action='store_false',
                        help='do not preserve the aspect ratio when resizing an image')

    parser.set_defaults(preserve_aspect_ratio=True)

    args = parser.parse_args()
    if args.lr_cosine_T_max is None:
        args.lr_cosine_T_max = args.epochs
    if not args.distributed:
        args.num_devices = 1

    return args 


if __name__ == '__main__':
    args = parse_args()
    if 'npu' in args.device:
        import torch.npu 
        if args.distributed:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '23333'

    main(args)
