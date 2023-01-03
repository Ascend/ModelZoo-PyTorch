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
from SFA3D.sfa.utils.misc import AverageMeter
from SFA3D.sfa.losses.losses import Compute_Loss
from SFA3D.sfa.utils.torch_utils import reduce_tensor, to_python_float
from SFA3D.sfa.test import parse_test_configs
from SFA3D.sfa.data_process.kitti_dataloader import create_val_dataloader
from SFA3D.sfa.models.model_utils import create_model


def validate(val_dataloader, model, configs):
    losses = AverageMeter('Loss', ':.4e')
    criterion = Compute_Loss(device=configs.device)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):
            metadatas, imgs, targets = batch_data
            batch_size = imgs.size(0)
            for k in targets.keys():
                targets[k] = targets[k].to(configs.device, non_blocking=True)
            imgs = imgs.to(configs.device, non_blocking=True).float()
            outputs = model(imgs)
            total_loss, loss_stats = criterion(outputs, targets)
            # For torch.nn.DataParallel case
            if (not configs.distributed) and (configs.gpu_idx is None):
                total_loss = torch.mean(total_loss)

            if configs.distributed:
                reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
            else:
                reduced_loss = total_loss.data
            losses.update(to_python_float(reduced_loss), batch_size)
            print(total_loss, loss_stats)
    print(losses.avg)


def load_pth_model(configs, checkpoint):
    model = create_model(configs)
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    model.eval()
    print('Loaded weights from {}'.format(checkpoint))
    return model

if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False
    configs.device = "cpu"
    configs.dataset_dir = './SFA3D/dataset/kitti'
    configs.pretrained_path = './SFA3D/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth'

    losses = AverageMeter('Loss', ':.4e')
    # load model, load val_dataset
    model = load_pth_model(configs, configs.pretrained_path)
    val_dataloader = create_val_dataloader(configs)

    validate(val_dataloader, model, configs)