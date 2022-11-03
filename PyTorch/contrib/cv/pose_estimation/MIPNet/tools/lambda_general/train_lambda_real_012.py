# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsLambdaMSELoss
from core.loss import JointsMSELoss
from core.train_general import train_lambda_012
from core.validate_general import validate_lambda_012

# from utils.utils import get_last_layer_optimizer
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_lambda_model_summary
from utils.utils import set_seed

import dataset
import models

# --------------------------------------------------------------------------------
set_seed(seed_id=0)

# --------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')


    args = parser.parse_args()

    return args
# --------------------------------------------------------------------------------

def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )    

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # copy train file
    shutil.copy2(
        __file__,
        final_output_dir)

    # copy synthetic dataset file
    shutil.copy2(
        os.path.join(this_dir, '../../lib/dataset', cfg.DATASET.SYNTHETIC_DATASET + '.py'),
        final_output_dir)


    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    dump_lambda = torch.rand(
        (1, 2)
    )
    ### this is used to visualize the network
    ### throws an assertion error on cube3, works well on bheem
    ### commented for now
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_lambda_model_summary(model, dump_input, dump_lambda))

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model_object = torch.load(cfg.TEST.MODEL_FILE)
        if 'latest_state_dict' in model_object.keys():
            logger.info('=> loading from latest_state_dict at {}'.format(cfg.TEST.MODEL_FILE))
            model.load_state_dict(model_object['latest_state_dict'], strict=False)
        else:
            logger.info('=> no latest_state_dict found')
            model.load_state_dict(model_object, strict=False)

    else:
        # print('error, please give model file')
        # exit()
        model_object = {}
        print('no model file give, initializing random')

    model = torch.nn.DataParallel(model).cuda()

     # ------------------------------------------
    # define loss function (criterion) and optimizer
    criterion_lambda = JointsLambdaMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.'+cfg.DATASET.TRAIN_DATASET)(
        cfg=cfg, image_dir=cfg.DATASET.TRAIN_IMAGE_DIR, annotation_file=cfg.DATASET.TRAIN_ANNOTATION_FILE, \
        dataset_type=cfg.DATASET.TRAIN_DATASET_TYPE, \
        image_set=cfg.DATASET.TRAIN_SET, is_train=True, \
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_dataset = eval('dataset.'+cfg.DATASET.TEST_DATASET)(
        cfg=cfg, image_dir=cfg.DATASET.TEST_IMAGE_DIR, annotation_file=cfg.DATASET.TEST_ANNOTATION_FILE, \
        dataset_type=cfg.DATASET.TEST_DATASET_TYPE, \
        image_set=cfg.DATASET.TEST_SET, is_train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    # # # ----------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False
    )

    # # # # # ---------------------------------------------
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    # # # # ----------------------------------------------
    ## resume optimizer
    if 'optimizer' in model_object.keys():
        logger.info('=> resuming optimizer from {}'.format(cfg.TEST.MODEL_FILE))
        optimizer.load_state_dict(model_object['optimizer'])

    cfg.defrost()
    resume_lr_step = []
    for lr_step in cfg.TRAIN.LR_STEP:
        lr_step = lr_step - begin_epoch
        if lr_step >= 0:
            resume_lr_step.append(lr_step)
    
    cfg.TRAIN.LR_STEP = resume_lr_step
    cfg.freeze()

    logger.info('=> updated lr schedule is {}'.format(cfg.TRAIN.LR_STEP))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    # -----------------------------------------------
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        # # # train for one epoch
        print('training on lambda for 0, 1, 2')
        train_lambda_012(cfg, train_loader, model, criterion_lambda, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, print_prefix='lambda_012')

        if epoch % cfg.EPOCH_EVAL_FREQ == 0:            
            validate_lambda_012(cfg, valid_loader, valid_dataset, model, criterion,
                     final_output_dir, tb_log_dir, writer_dict, epoch=epoch, print_prefix='lambda_012')

        lr_scheduler.step()
        perf_indicator = 0.0

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'latest_state_dict': model.module.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint_{}.pth'.format(epoch + 1))

    # # ----------------------------------------------
    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()