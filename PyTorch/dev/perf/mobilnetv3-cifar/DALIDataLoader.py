'''
A new dataloader using NVIDIA DALI in order to speed up the dataloader in pytorch
Ref:    https://github.com/d-li14/mobilenetv2.pytorch/blob/master/utils/dataloaders.py
        https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
'''

import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from math import ceil

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    print("Please install DALI from https://www.github.com/NVIDIA/DALI to run DataLoader.")


class TinyImageNetHybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, seed, dali_cpu=False):
        super(TinyImageNetHybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed)
    
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1
        self.input = ops.FileReader(
                file_root=data_dir,
                shard_id=local_rank,
                num_shards=world_size,
                pad_last_batch=True,
                random_shuffle=False,
                shuffle_after_epoch=True)
        
        # decide to work on cpu or gpu
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'

        self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device=dali_device, size=crop, random_aspect_ratio=[0.75, 4./3],
                                         random_area=[0.08, 1.0], num_attempts=100, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device='gpu',
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485*255, 0.456*255, 0.406*255],
                                            std=[0.229*255, 0.224*255, 0.225*255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name='Reader')
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror = rng)
        return [output, self.labels]


class TinyImageNetHybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, seed):
        super(TinyImageNetHybridValPipe, self).__init__(batch_size, num_threads, device_id, seed)

        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1
        self.input = ops.FileReader(
                file_root=data_dir,
                shard_id=local_rank,
                num_shards=world_size,
                pad_last_batch=True,
                random_shuffle=False)

        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.cmnp = ops.CropMirrorNormalize(device='gpu',
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485*255, 0.456*255, 0.406*255],
                                            std=[0.229*255, 0.224*255, 0.225*255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name='Reader')
        images = self.decode(self.jpegs)
        output = self.cmnp(images)
        return [output, self.labels]

class ImageNetHybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, seed, dali_cpu=False):
        super(ImageNetHybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = seed)
    
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1
        self.input = ops.FileReader(
                file_root=data_dir,
                shard_id=local_rank,
                num_shards=world_size,
                pad_last_batch=True,
                random_shuffle=False,
                shuffle_after_epoch=True)
        
        # decide to work on cpu or gpu
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'

        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        '''
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.75, 1.25],
                                                 random_area=[0.08, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        '''
        self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB,
                                       device_memory_padding=device_memory_padding,
                                       host_memory_padding=host_memory_padding,)
        self.res = ops.RandomResizedCrop(device=dali_device, size=crop, random_aspect_ratio=[0.75, 4./3],
                                         random_area=[0.08, 1.0], num_attempts=100, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device='gpu',
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name='Reader')
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror = rng)
        return [output, self.labels]


class ImageNetHybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, seed):
        super(ImageNetHybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = seed)

        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1
        self.input = ops.FileReader(
                file_root=data_dir,
                shard_id=local_rank,
                num_shards=world_size,
                pad_last_batch=True,
                random_shuffle=False)

        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.res = ops.Resize(device='gpu', resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device='gpu',
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485*255, 0.456*255, 0.406*255],
                                            std=[0.229*255, 0.224*255, 0.225*255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name='Reader')
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

class DALIWrapper(object):
    def gen_wrapper(dali_pipeline):
        for data in dali_pipeline:
            input = data[0]['data']
            target = data[0]['label'].squeeze().cuda().long()
            yield input, target

    def __init__(self, dali_pipeline):
        self.dali_pipeline = dali_pipeline

    def __iter__(self):
        return DALIWrapper.gen_wrapper(self.dali_pipeline)

def get_dali_tinyImageNet_train_loader(data_path, batch_size, seed, num_threads=4, dali_cpu=False):
    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        world_size = 1
        
    train_dir = os.path.join(data_path, 'train')
    
    pipe = TinyImageNetHybridTrainPipe(batch_size=batch_size, num_threads=num_threads,
                           device_id=local_rank, data_dir=train_dir,
                           crop=56, seed=seed, dali_cpu=dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size('Reader') / world_size), fill_last_batch=False, last_batch_padded=True, auto_reset=True)
    
    return DALIWrapper(train_loader), ceil(pipe.epoch_size('Reader') / (world_size*batch_size))


def get_dali_tinyImageNet_val_loader(data_path, batch_size, seed, num_threads=4):
    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        world_size = 1
    val_dir = os.path.join(data_path, 'val')
    
    pipe = TinyImageNetHybridValPipe(batch_size=batch_size, num_threads=num_threads,
                         device_id=local_rank, data_dir=val_dir,
                         crop=56, seed=seed)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size('Reader')/world_size), fill_last_batch=False, last_batch_padded=True, auto_reset=True)
    
    return DALIWrapper(val_loader), ceil(pipe.epoch_size('Reader') / (world_size * batch_size))

def get_dali_imageNet_train_loader(data_path, batch_size, seed, num_threads=4, dali_cpu=False):
    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        world_size = 1
        
    train_dir = os.path.join(data_path, 'ILSVRC2012_img_train')
    
    pipe = ImageNetHybridTrainPipe(batch_size=batch_size, num_threads=num_threads,
                           device_id=local_rank, data_dir=train_dir,
                           crop=224, seed=seed, dali_cpu=dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size('Reader') / world_size), fill_last_batch=False, last_batch_padded=True, auto_reset=True)
    
    return DALIWrapper(train_loader), ceil(pipe.epoch_size('Reader') / (world_size*batch_size))


def get_dali_imageNet_val_loader(data_path, batch_size, seed, num_threads=4):
    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        world_size = 1
    val_dir = os.path.join(data_path, 'ILSVRC2012_img_val')
    
    pipe = ImageNetHybridValPipe(batch_size=batch_size, num_threads=num_threads,
                         device_id=local_rank, data_dir=val_dir,
                         crop=224, size=256, seed=seed)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size('Reader')/world_size), fill_last_batch=False, last_batch_padded=True, auto_reset=True)
    
    return DALIWrapper(val_loader), ceil(pipe.epoch_size('Reader') / (world_size * batch_size))