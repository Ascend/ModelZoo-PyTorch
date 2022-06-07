#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=256, 
        training=False,
        num_workers=12
    )

    # build model architecture
    if 'returns_feat' in config['arch']['args']:
        model = config.init_obj('arch', module_arch, allow_override=True, returns_feat=False)
    else:
        model = config.init_obj('arch', module_arch) 

    # get function handles of loss and metrics 
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
    model = model.to(f'npu:{NPU_CALCULATE_DEVICE}')
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    num_classes = config._config["arch"]["args"]["num_classes"]
    confusion_matrix = torch.zeros(num_classes, num_classes).npu()

    get_class_acc = True
    if get_class_acc:
        train_data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=256,
            training=True
        ) 
        train_cls_num_list = np.array(train_data_loader.cls_num_list)
        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
        few_shot = train_cls_num_list < 20

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(f'npu:{NPU_CALCULATE_DEVICE}'), target.to(f'npu:{NPU_CALCULATE_DEVICE}')
            output = model(data)
 
            batch_size = data.shape[0] 
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

            for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
 

    acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
    
    acc = acc_per_class.cpu().numpy() 
  
    many_shot_acc = acc[many_shot].mean()
    medium_shot_acc = acc[medium_shot].mean()
    few_shot_acc = acc[few_shot].mean() 

    n_samples = len(data_loader.sampler)
    log = {}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })

    if get_class_acc:
        log.update({
            "many_class_num": many_shot.sum(),
            "medium_class_num": medium_shot.sum(),
            "few_class_num": few_shot.sum(),
            "many_shot_acc": many_shot_acc,
            "medium_shot_acc": medium_shot_acc,
            "few_shot_acc": few_shot_acc,
        })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
