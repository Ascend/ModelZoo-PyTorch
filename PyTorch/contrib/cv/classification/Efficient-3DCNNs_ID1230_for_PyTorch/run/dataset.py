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
from datasets.ucf101 import UCF101
from run.spatial_transforms import *
from run.temporal_transforms import *
from run.target_transforms import ClassLabel, VideoID
from torch.utils.data.distributed import DistributedSampler

def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['ucf101']
    training_data = UCF101(
        opt.video_path,
        opt.annotation_path,
        'training',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration)
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['ucf101']
    validation_data = UCF101(
        opt.video_path,
        opt.annotation_path,
        'validation',
        opt.n_val_samples,
        spatial_transform,
        temporal_transform,
        target_transform,
        sample_duration=opt.sample_duration)
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['ucf101']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'

    test_data = UCF101(
        opt.video_path,
        opt.annotation_path,
        subset,
        0,
        spatial_transform,
        temporal_transform,
        target_transform,
        sample_duration=opt.sample_duration)
    return test_data


def get_norm_method(opt):
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    return norm_method

def get_train_loader(opt):
    norm_method = get_norm_method(opt)
    assert opt.train_crop in ['random', 'corner', 'center']
    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c'])
    spatial_transform = Compose([
        RandomHorizontalFlip(),
        # RandomRotate(),
        # RandomResize(),
        crop_method,
        # MultiplyValues(),
        # Dropout(),
        # SaltImage(),
        # Gaussian_blur(),
        # SpatialElasticDisplacement(),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
    target_transform = ClassLabel()

    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)

    if opt.gpu_or_npu == 'npu':
        if opt.distributed:
            train_sampler = DistributedSampler(training_data)
            train_loader = torch.utils.data.DataLoader(training_data,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.n_threads,
                            sampler=train_sampler,
                            pin_memory=False,
                            drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(training_data,
                           batch_size=opt.batch_size,
                           shuffle=True,
                           num_workers=opt.n_threads,
                           pin_memory=False,
                           drop_last=True)
    elif opt.gpu_or_npu == 'gpu':
        if opt.distributed:
            train_sampler = DistributedSampler(training_data)
            train_loader = torch.utils.data.DataLoader(training_data,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.n_threads,
                            sampler=train_sampler,
                            pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(training_data,
                           batch_size=opt.batch_size,
                           shuffle=True,
                           num_workers=opt.n_threads,
                           pin_memory=True)
    return train_loader


def get_val_loader(opt):
    norm_method = get_norm_method(opt)
    spatial_transform = Compose(
        [Scale(opt.sample_size), CenterCrop(opt.sample_size), ToTensor(opt.norm_value), norm_method])
    # temporal_transform = LoopPadding(opt.sample_duration)
    temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)

    if opt.gpu_or_npu == 'npu':
        val_loader = torch.utils.data.DataLoader(validation_data,
                     batch_size=opt.batch_size,
                     shuffle=True,
                     num_workers=opt.n_threads,
                     pin_memory=False,
                     drop_last=True)
    elif opt.gpu_or_npu == 'gpu':
        val_loader = torch.utils.data.DataLoader(validation_data,
                     batch_size=opt.batch_size,
                     shuffle=True,
                     num_workers=opt.n_threads,
                     pin_memory=True)
    return val_loader

def get_test_loader(opt):
    norm_method = get_norm_method(opt)

    spatial_transform = Compose([
        Scale(int(opt.sample_size / opt.scale_in_test)),
        CornerCrop(opt.sample_size, opt.crop_position_in_test),
        ToTensor(opt.norm_value), norm_method
    ])
    # temporal_transform = LoopPadding(opt.sample_duration, opt.downsample)
    temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
    target_transform = VideoID()

    test_data = get_test_set(opt, spatial_transform, temporal_transform,
                             target_transform)

    if opt.gpu_or_npu == 'npu':
        test_loader = torch.utils.data.DataLoader(test_data,
                     batch_size=16,
                     shuffle=False,
                     num_workers=opt.n_threads,
                     pin_memory=False,
                     drop_last=True)
    elif opt.gpu_or_npu == 'gpu':
        test_loader = torch.utils.data.DataLoader(test_data,
            batch_size=16,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
    return test_data, test_loader