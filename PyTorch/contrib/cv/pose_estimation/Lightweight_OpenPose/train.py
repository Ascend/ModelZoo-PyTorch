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
import os
import shutil
import time
import cv2
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import apex
from apex import amp
from datasets.coco import CocoTrainDataset
from datasets.transformations import ConvertKeypoints, Scale, Rotate, CropPad, Flip
from modules.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.loss import l2_loss, AverageMeter, ProgressMeter
from modules.load_state import load_state, load_from_mobilenet
from val import evaluate
from multi_epochs_dataloaders import MultiEpochsDataLoader
try:
    from torch_npu.utils.profiler import Profile
except:
    print("Profile not in torch_npu.utils.profiler now..Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def end(self):
            pass

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoaderl

parser = argparse.ArgumentParser()
# openpose parameters
# for dataset
parser.add_argument('--prepared-train-labels', type=str, required=True,
                    help='path to the file with prepared annotations')
parser.add_argument('--train-images-folder', type=str, required=True,
                    help='path to COCO train images folder')
parser.add_argument('--val-labels', type=str, required=True,
                    help='path to json with keypoints val labels')
parser.add_argument('--val-images-folder', type=str, required=True,
                    help='path to COCO val images folder')
parser.add_argument('--num-workers', type=int, default=8,
                    help='number of workers')
# for model
parser.add_argument('--num-refinement-stages', type=int, default=1,
                    help='number of refinement stages')
# for train
parser.add_argument('--epochs', type=int, default=280,
                    help='epochs')
parser.add_argument('--batch-size', type=int, default=80,
                    help='batch size')
parser.add_argument('--base-lr', type=float, default=4e-5,
                    help='initial learning rate')
parser.add_argument('--checkpoint-path', type=str, required=True,
                    help='path to the checkpoint to continue training from')
parser.add_argument('--from-mobilenet', action='store_true',
                    help='load weights from mobilenet feature extractor')
parser.add_argument('--weights-only', action='store_true',
                    help='just initialize layers with pre-trained weights and start training from the beginning')
# for save,log,val
parser.add_argument('--experiment-name', type=str, default='default',
                    help='experiment name to create folder for checkpoints')
parser.add_argument('--val-output-name', type=str, default='detections.json',
                    help='name of output json file with detected keypoints')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default 10)')
parser.add_argument('--eval-freq', default=5, type=int, metavar='N',
                    help='evaluate frequency (default 10)')
# add required parameters
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                    help="eval openpose model")
parser.add_argument('--prof', default=False, action='store_true',
                    help='use profiling to evaluate the performance of model')
# parameters for distribute training
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
# add parameters for  ascend 910
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--addr', default='10.136.181.115',
                    type=str, help='master addr')
parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7',
                    type=str, help='device id list')
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=None, type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')

best_ap = 0


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def get_device_name(device_type, device_order):
    if device_type == 'npu':
        device_name = 'npu:{}'.format(device_order)
    else:
        device_name = 'cuda:{}'.format(device_order)

    return device_name


def main():
    args = parser.parse_args()

    # add checkpoints_folder
    checkpoints_folder = '{}_checkpoints'.format(args.experiment_name)
    print("checkpoint_folder : ", checkpoints_folder)
    if not os.path.exists(checkpoints_folder) and not args.evaluate:
        os.makedirs(checkpoints_folder)
    args.checkpoints_folder = checkpoints_folder
    args.process_device_map = device_id_to_process_device_map(args.device_list)

    # add start_epoch
    args.start_epoch = 0
    print(args.device_list)

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29688'

    if args.device == 'npu':
        ngpus_per_node = len(args.process_device_map)
    else:
        if args.gpu is None:
            ngpus_per_node = len(args.process_device_map)
        else:
            ngpus_per_node = 1
    print('ngpus_per_node:', ngpus_per_node)

    args.world_size = ngpus_per_node * args.world_size
    args.distributed = args.world_size > 1

    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_ap
    if args.distributed:
        args.gpu = args.process_device_map[gpu]

    args.rank = args.rank * ngpus_per_node + gpu
    if args.distributed:
        if args.device == 'npu':
            dist.init_process_group(backend=args.dist_backend,
                                    # init_method=args.dist_url,
                                    world_size=args.world_size,
                                    rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend,
                                    init_method=args.dist_url,
                                    world_size=args.world_size,
                                    rank=args.rank)
    print('rank: {} / {}'.format(args.rank, args.world_size))

    # init device
    device_loc = get_device_name(args.device, args.gpu)
    args.loc = device_loc

    # set device
    print('set_device ', device_loc)
    if args.device == 'npu':
        torch.npu.set_device(device_loc)
    else:
        torch.cuda.set_device(args.gpu)

    # create model
    model = PoseEstimationWithMobileNet(1 if args.evaluate else args.num_refinement_stages)
    # load checkpoint
    if args.checkpoint_path:
        print("=> using pre-trained model", " device(%d)," % args.gpu)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

        if args.from_mobilenet:
            print("loading model of yours...(mobilenet)", " device(%d)," % args.gpu)
            load_from_mobilenet(model, checkpoint)
        else:
            print("loading model of yours...(trained model)", " device(%d)," % args.gpu)
            if 'ap' in checkpoint:
                best_ap = checkpoint['ap']
                print("previous best ap : {:.2f}".format(best_ap * 100), " device(%d)," % args.gpu)
            load_state(model, checkpoint)
    else:
        print("=> creating model openpose", " device(%d)," % args.gpu)

    print('model to device_loc(%s)...' % device_loc)
    model = model.to(device_loc)

    # eval dataset
    args.val_output_name = "device({})".format(args.gpu) + args.val_output_name

    if args.distributed:
        args.batch_size = int(args.batch_size / args.world_size)
        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    # define optimizer, apply apex
    optimizer = apex.optimizers.NpuFusedAdam([
        {'params': get_parameters_conv(model.model, 'weight')},
        {'params': get_parameters_conv_depthwise(model.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(model.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(model.model, 'bias'), 'lr': args.base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(model.cpm, 'weight'), 'lr': args.base_lr},
        {'params': get_parameters_conv(model.cpm, 'bias'), 'lr': args.base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(model.cpm, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv(model.initial_stage, 'weight'), 'lr': args.base_lr},
        {'params': get_parameters_conv(model.initial_stage, 'bias'), 'lr': args.base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(model.refinement_stages, 'weight'), 'lr': args.base_lr * 4},
        {'params': get_parameters_conv(model.refinement_stages, 'bias'), 'lr': args.base_lr * 8, 'weight_decay': 0},
        {'params': get_parameters_bn(model.refinement_stages, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(model.refinement_stages, 'bias'), 'lr': args.base_lr * 2, 'weight_decay': 0},
    ], lr=args.base_lr, weight_decay=5e-4)

    if args.amp:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale, combine_grad=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 260], gamma=0.333)

    if args.checkpoint_path and not args.from_mobilenet and not args.weights_only:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.checkpoint_path, checkpoint['epoch']), " device(%d)," % args.gpu)
        if args.amp:
            amp.load_state_dict(checkpoint['amp'])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    cudnn.benchmark = True

    # train dataset
    train_dataset = CocoTrainDataset(args.prepared_train_labels, args.train_images_folder,
                                     stride=8, sigma=7, paf_thickness=1,
                                     transform=transforms.Compose([
                                         ConvertKeypoints(),
                                         Scale(),
                                         Rotate(pad=(128, 128, 128)),
                                         CropPad(pad=(128, 128, 128)),
                                         Flip()]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataloader_fn = MultiEpochsDataLoader  # torch.utils.data.DataLoader
    train_loader = dataloader_fn(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if args.evaluate:
        print("evaluate mode...", " device(%d)," % args.gpu)
        evaluate(args.val_labels, args.val_output_name, args.val_images_folder, model, args=args)
        return

    if args.prof:
        print("profiling mode...", " device(%d)," % args.gpu)
        profiling(train_loader, model, optimizer, args)
        return

    print("train mode...", " device(%d)," % args.gpu)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args, ngpus_per_node)
        scheduler.step()

        if (epoch + 1) % args.eval_freq == 0:
            # evaluate on validation set
            print("device({}), validation, current epoch : {} ...".format(args.gpu, epoch))
            ap = evaluate(args.val_labels,
                          args.val_output_name,
                          args.val_images_folder,
                          model,
                          args=args)

            is_best = ap > best_ap
            best_ap = max(ap, best_ap)
            if not args.distributed or (args.distributed and args.gpu == args.process_device_map[0]):
                print("device(0),epoch({}),ap:{:.4f}".format(epoch, ap))
                if is_best:
                    print("best model get... ap :%.4f" % ap)
                ############## npu modify begin #############
                if args.amp:
                    snapshot_name = '{}/checkpoint_amp_{}.pth'.format(args.checkpoints_folder, epoch)
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': 'openpose',
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'ap': best_ap
                    }, is_best, snapshot_name, ap)
                    snapshot_name = '{}/checkpoint_amp_{}.pth'.format(args.checkpoints_folder, epoch - args.eval_freq)
                else:
                    snapshot_name = '{}/checkpoint_{}.pth'.format(args.checkpoints_folder, epoch)
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': 'openpose',
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'ap': best_ap
                    }, is_best, snapshot_name, ap)
                    snapshot_name = '{}/checkpoint_{}.pth'.format(args.checkpoints_folder, epoch - args.eval_freq)
                if os.path.exists(snapshot_name):
                    os.remove(snapshot_name)
            ############## npu modify end #############


def train(train_loader, model, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    meter_list = [batch_time, data_time]
    for idx in range(args.num_refinement_stages + 1):
        meter_list.append(AverageMeter("s%d_paf" % (idx + 1), ':.4e'))
        meter_list.append(AverageMeter("s%d_heatmap" % (idx + 1), ':.4e'))
    progress = ProgressMeter(len(train_loader), meter_list, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    profile = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                      profile_type=os.getenv('PROFILE_TYPE'))

    for i, batch_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = batch_data['image']
        keypoint_masks = batch_data['keypoint_mask']
        paf_masks = batch_data['paf_mask']
        keypoint_maps = batch_data['keypoint_maps']
        paf_maps = batch_data['paf_maps']

        # Configure input
        images = images.to(args.loc, non_blocking=True).to(torch.float)
        keypoint_masks = keypoint_masks.to(args.loc, non_blocking=True).to(torch.float)
        paf_masks = paf_masks.to(args.loc, non_blocking=True).to(torch.float)
        keypoint_maps = keypoint_maps.to(args.loc, non_blocking=True).to(torch.float)
        paf_maps = paf_maps.to(args.loc, non_blocking=True).to(torch.float)

        profile.start()
        stages_output = model(images)

        losses = []
        for loss_idx in range(args.num_refinement_stages + 1):
            losses.append(l2_loss(stages_output[loss_idx * 2], keypoint_maps, keypoint_masks, images.shape[0]))
            losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))
            meter_list[(loss_idx + 1) * 2 + 1].update(losses[-1].item(), images.shape[0])
            meter_list[(loss_idx + 1) * 2].update(losses[-2].item(), images.shape[0])
        loss = sum(losses)

        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        profile.end()

        # measure elapsed time
        cost_time = time.time() - end
        batch_time.update(cost_time)
        end = time.time()

        if not args.distributed or (args.distributed and args.gpu == args.process_device_map[0]):
            if i % args.print_freq == 0:
                progress.display(i)
            if batch_time.avg:
                print("[npu id:", args.gpu, "]",
                      "batch_size:", args.world_size * args.batch_size,
                      'Time: {:.3f}'.format(batch_time.avg),
                      '* FPS@all {:.3f}'.format(args.batch_size * args.world_size / batch_time.avg))
    # train loop done


def profiling(data_loader, model, optimizer, args):
    # switch to train mode
    model.train()

    def update(step=None):
        start_time = time.time()
        stages_output = model(images)

        losses = []
        for loss_idx in range(args.num_refinement_stages + 1):
            losses.append(l2_loss(stages_output[loss_idx * 2], keypoint_maps, keypoint_masks, images.shape[0]))
            losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))

        loss = sum(losses)
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        if step is not None:
            print('iter: %d, loss: %.2f, time: %.2f' % (step, loss, (time.time() - start_time)))

    for step, batch_data in enumerate(data_loader):
        images = batch_data['image']
        keypoint_masks = batch_data['keypoint_mask']
        paf_masks = batch_data['paf_mask']
        keypoint_maps = batch_data['keypoint_maps']
        paf_maps = batch_data['paf_maps']

        # Configure input
        images = images.to(args.loc, non_blocking=True).to(torch.float)
        keypoint_masks = keypoint_masks.to(args.loc, non_blocking=True).to(torch.float)
        paf_masks = paf_masks.to(args.loc, non_blocking=True).to(torch.float)
        keypoint_maps = keypoint_maps.to(args.loc, non_blocking=True).to(torch.float)
        paf_maps = paf_maps.to(args.loc, non_blocking=True).to(torch.float)

        if step < 20:
            update(step)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update()
            else:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update()
            break

    prof.export_chrome_trace("%s.prof" % args.device)


def save_checkpoint(state, is_best, filename='checkpoint.pth', ap=0):
    torch.save(state, filename)
    if is_best:
        print("current best ap : {:.2f}".format(ap * 100))
        shutil.copyfile(filename, 'model_best.pth')


if __name__ == '__main__':
    main()
