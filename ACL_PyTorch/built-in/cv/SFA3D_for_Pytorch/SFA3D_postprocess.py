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


import torch
import numpy as np
import argparse
from easydict import EasyDict as edict
from SFA3D.sfa.utils.misc import AverageMeter
from SFA3D.sfa.losses.losses import Compute_Loss
from SFA3D.sfa.utils.torch_utils import reduce_tensor, to_python_float
from SFA3D.sfa.data_process.kitti_dataloader import create_val_dataloader


def parse_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--dataset_dir', type=str,
                        default='./SFA3D/dataset/kitti', metavar='PATH',
                        help='the path of the KITTI dataset')
    parser.add_argument('--result_path', type=str,
                        default='./ais_infer_result/dumpdata_outputs', metavar='PATH',
                        help='the path of the inference result')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.dowm_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4
    return configs


def get_outputs(idx, result_path):
    outputs = {}
    outputs['hm_cen'] = torch.from_numpy(
        np.fromfile(result_path + idx + '_0.bin', dtype=np.float32).reshape(1, -1, 152, 152))
    outputs['cen_offset'] = torch.from_numpy(
        np.fromfile(result_path + idx + '_1.bin', dtype=np.float32).reshape(1, -1, 152, 152))
    outputs['direction'] = torch.from_numpy(
        np.fromfile(result_path + idx + '_2.bin', dtype=np.float32).reshape(1, -1, 152, 152))
    outputs['z_coor'] = torch.from_numpy(
        np.fromfile(result_path + idx + '_3.bin', dtype=np.float32).reshape(1, -1, 152, 152))
    outputs['dim'] = torch.from_numpy(
        np.fromfile(result_path + idx + '_4.bin', dtype=np.float32).reshape(1, -1, 152, 152))
    return outputs


def get_losses():
    configs = parse_configs()
    configs.distributed = False  # val testing
    configs.device = "cpu"

    losses = AverageMeter('Loss', ':.4e')
    val_dataloader = create_val_dataloader(configs)
    criterion = Compute_Loss(device=configs.device)

    for batch_idx, batch_data in enumerate(val_dataloader):
        idx = '{:06d}'.format(batch_idx + 6000)  # val id: 006000~007480
        metadatas, imgs, targets = batch_data
        batch_size = imgs.size(0)
        for k in targets.keys():
            targets[k] = targets[k].to(configs.device, non_blocking=True)
        outputs = get_outputs(idx, configs.result_path)

        total_loss, loss_stats = criterion(outputs, targets)
        if (not configs.distributed) and (configs.gpu_idx is None):
            total_loss = torch.mean(total_loss)
        if configs.distributed:
            reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
        else:
            reduced_loss = total_loss.data
        losses.update(to_python_float(reduced_loss), batch_size)
        print(idx, total_loss, loss_stats)
    print('total_loss_avg: ', losses.avg)
    return losses.avg


if __name__ == '__main__':
    get_losses()