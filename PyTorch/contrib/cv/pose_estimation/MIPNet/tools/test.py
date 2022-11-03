# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse  #命令项选项与参数解析的模块
import os
import pprint

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
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from utils.utils import get_model_summary


import dataset
import models


def parse_args():
    # 1.创建解析器
    parser = argparse.ArgumentParser(description='Train keypoints network') 
    # 2.general 添加参数，及参数信息
    # help：帮助信息； nargs：应读取参数个数；
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

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

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args() #3.解析参数 1 namespace对象按解析出的属性构建
    return args


def main():
    args = parse_args()
    update_config(cfg, args) 

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid') #日志(cfg, cfg_name, phase='train')

    logger.info(pprint.pformat(args)) #打印arg内容
    logger.info(cfg) #打印cfg内容

    # cudnn related setting 参数传递
    cudnn.benchmark = cfg.CUDNN.BENCHMARK  #=true；自动寻找高效算法
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC #=false；不采用默认算法
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED #=true：使用非确定性算法
    # get_pose_net(cfg, is_train, **kwargs):
    # model = PoseHighResolutionNet(cfg, **kwargs) HRNet
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )  

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )# 随机生成模型初始化

    logger.info(get_model_summary(model, dump_input))  #展示网络计算量和参数量

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE)) #加载model
        model_object = torch.load(cfg.TEST.MODEL_FILE)
        if 'latest_state_dict' in model_object.keys():
            logger.info('=> loading from latest_state_dict at {}'.format(cfg.TEST.MODEL_FILE))
            model.load_state_dict(model_object['latest_state_dict'], strict=False)
        else:
            logger.info('=> no latest_state_dict found')
            model.load_state_dict(model_object, strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).npu()
    #model = torch.nn.DataParallel(model).npu() #多GPU加速
    model = model.npu()

    # define loss function (criterion) and optimizer；返回预测热力图和target的平均损失
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).npu() 

    # Data loading code #数据归一化
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ) 
    #数据集
    valid_dataset = eval('dataset.'+cfg.DATASET.TEST_DATASET)(
        cfg=cfg, image_dir=cfg.DATASET.TEST_IMAGE_DIR, annotation_file=cfg.DATASET.TEST_ANNOTATION_FILE, \
        dataset_type=cfg.DATASET.TEST_DATASET_TYPE, \
        image_set=cfg.DATASET.TEST_SET, is_train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False, #打乱数据
        num_workers=cfg.WORKERS, #子进程数
        pin_memory=True  #内存寄存
    )

    # evaluate on validation set 验证集评估
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir, writer_dict)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
