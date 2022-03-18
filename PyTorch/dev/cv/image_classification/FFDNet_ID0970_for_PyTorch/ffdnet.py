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
import numpy as np
import cv2
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import FFDNet
import utils
try:
    from apex import amp
except ImportError:
    amp = None
import apex

def read_image(image_path, is_gray):
    """
    :return: Normalized Image (C * W * H)
    """
    if is_gray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image.T, 0) # 1 * W * H
    else:
        image = cv2.imread(image_path)
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).transpose(2, 1, 0) # 3 * W * H
    
    return utils.normalize(image)

def load_images(is_train, is_gray, base_path):
    """
    :param base_path: ./train_data/
    :return: List[Patches] (C * W * H)
    """
    if is_gray:
        train_dir = 'gray/train/'
        val_dir = 'gray/val/'
    else:
        #train_dir = 'rgb/train/'
        train_dir = 'rgb/'
        val_dir = 'rgb/val/'
    
    image_dir = base_path.replace('\'', '').replace('"', '') + (train_dir if is_train else val_dir)
    print('> Loading images in ' + image_dir)
    images = []
    for fn in next(os.walk(image_dir))[2]:
        image = read_image(image_dir + fn, is_gray)
        images.append(image)
    return images

def images_to_patches(images, patch_size):
    """
    :param images: List[Image (C * W * H)]
    :param patch_size: int
    :return: (n * C * W * H)
    """
    patches_list = []
    for image in images:
        patches = utils.image_to_patches(image, patch_size=patch_size)
        if len(patches) != 0:
            patches_list.append(patches)
    del images
    return np.vstack(patches_list)

def train(args):
    print('> Loading dataset...')
    # Images
    train_dataset = load_images(is_train=True, is_gray=args.is_gray, base_path=args.train_path)
    val_dataset = load_images(is_train=False, is_gray=args.is_gray, base_path=args.train_path)
    print(f'\tTrain image datasets: {len(train_dataset)}')
    print(f'\tVal image datasets: {len(val_dataset)}')

    # Patches
    train_dataset = images_to_patches(train_dataset, patch_size=args.patch_size)
    val_dataset = images_to_patches(val_dataset, patch_size=args.patch_size)
    print(f'\tTrain patch datasets: {train_dataset.shape}')
    print(f'\tVal patch datasets: {val_dataset.shape}')

    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=96, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=96, pin_memory=True)
    print(f'\tTrain batch number: {len(train_dataloader)}')
    print(f'\tVal batch number: {len(val_dataloader)}')

    # Noise list
    train_noises = args.train_noise_interval # [0, 75, 15]
    val_noises = args.val_noise_interval # [0, 60, 30]
    train_noises = list(range(train_noises[0], train_noises[1], train_noises[2]))
    val_noises = list(range(val_noises[0], val_noises[1], val_noises[2]))
    print(f'\tTrain noise internal: {train_noises}')
    print(f'\tVal noise internal: {val_noises}')
    print('\n')

    # Model & Optim
    model = FFDNet(is_gray=args.is_gray)
    model.apply(utils.weights_init_kaiming)
    #if args.cuda:
    if args.npu:
        #model = model.cuda()
        model = model.to('npu', non_blocking=True)
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=args.learning_rate)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                               opt_level=args.apex_opt_level,
                                               loss_scale=args.loss_scale_value,
                                               combine_grad=True)

    print('> Start training...')
    data_prefetcher_stream = torch.npu.Stream()
    from prefetcher import Prefetcher
    for epoch_idx in range(args.epoches):
        # Train
        loss_idx = 0
        train_losses = 0
        model.train()

        start_time = time.time()
        n = 0
        batch_idx = 0
        prefetcher = Prefetcher(train_dataloader, train_noises, stream=data_prefetcher_stream)
        batch_data, new_images, noise_sigma = prefetcher.next()
        
        while batch_data is not None:
            batch_idx += 1
            n += 1
            if n == 150:
                break  #涓嶈窇鍐掔儫娉ㄩ噴姝よ
                #pass  #璺戝啋鐑熸敞閲婃琛
            i = 0
            # According to internal, add noise
            for int_noise_sigma in train_noises:
                start_time = time.time()
                # Predict
                images_pred = model(new_images[i], noise_sigma[i])
                #train_loss = loss_fn(images_pred, batch_data.to(images_pred.device))
                train_loss = loss_fn(images_pred, batch_data)
                train_losses += train_loss
                loss_idx += 1

                optimizer.zero_grad()
                if args.apex:
                    with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    train_loss.backward()
                #train_loss.backward()
                optimizer.step()
                i = i + 1
                # Log Progress
                stop_time = time.time()
                all_num = len(train_dataloader) * len(train_noises)
                done_num = batch_idx * len(train_noises) + train_noises.index(int_noise_sigma) + 1
                rest_time = int((stop_time - start_time) / done_num * (all_num - done_num))
                percent = int(done_num / all_num * 100)
                fps = args.batch_size / (stop_time - start_time)
                print(f'\rEpoch: {epoch_idx + 1} / {args.epoches}, ' +
                      f'Batch: {batch_idx + 1} / {len(train_dataloader)}, ' +
                      f'Noise_Sigma: {int_noise_sigma} / {train_noises[-1]}, ' +
                      f'Train_Loss: {train_loss}, ' +
                      f'=> {rest_time}s, {percent}%' + f'fps: {fps}')
            batch_data, new_images, noise_sigma = prefetcher.next()
        train_losses /= loss_idx
        print(f', Avg_Train_Loss: {train_losses}, All: {int(stop_time - start_time)}s')
        
        # Evaluate
        loss_idx = 0
        val_losses = 0
        if (epoch_idx + 1) % args.val_epoch != 0:
            continue
        model.eval()
        
        start_time = time.time()
        for batch_idx, batch_data in enumerate(val_dataloader):
            # According to internal, add noise
            for int_noise_sigma in val_noises:
                noise_sigma = int_noise_sigma / 255
                new_images = utils.add_batch_noise(batch_data, noise_sigma)
                noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(new_images.shape[0])]))
                new_images = Variable(new_images)
                noise_sigma = Variable(noise_sigma)
                #if args.cuda:
                if args.npu:
                    #new_images = new_images.cuda()
                    new_images = new_images.npu()
                    #noise_sigma = noise_sigma.cuda()
                    noise_sigma = noise_sigma.npu()

                # Predict
                images_pred = model(new_images, noise_sigma)
                #val_loss = loss_fn(images_pred, batch_data.to(images_pred.device))
                val_loss = loss_fn(images_pred, batch_data.to('npu'))
                val_losses += val_loss
                loss_idx += 1
                
                # Log Progress
                stop_time = time.time()
                all_num = len(val_dataloader) * len(val_noises)
                done_num = batch_idx * len(val_noises) + val_noises.index(int_noise_sigma) + 1
                rest_time = int((stop_time - start_time) / done_num * (all_num - done_num))
                percent = int(done_num / all_num * 100)
                print(f'Epoch: {epoch_idx + 1} / {args.epoches}, ' +
                      f'Batch: {batch_idx + 1} / {len(val_dataloader)}, ' +
                      f'Noise_Sigma: {int_noise_sigma} / {val_noises[-1]}, ' +
                      f'Val_Loss: {val_loss}, ' + 
                      f'=> {rest_time}s {percent}%')
                
        val_losses /= loss_idx
        print(f', Avg_Val_Loss: {val_losses}, All: {int(stop_time - start_time)}s')

        # Save Checkpoint
        if (epoch_idx + 1) % args.save_checkpoints == 0:
            model_path = args.model_path + ('net_gray_checkpoint.pth' if args.is_gray else 'net_rgb_checkpoint.pth')
            torch.save(model.state_dict(), model_path)
            print(f'| Saved Checkpoint at Epoch {epoch_idx + 1} to {model_path}')

    # Final Save Model Dict
    model.eval()
    model_path = args.model_path + ('net_gray.pth' if args.is_gray else 'net_rgb.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Saved State Dict in {model_path}')
    print('\n')

def test(args):
    # Image
    image = cv2.imread(args.test_path)
    if image is None:
        raise Exception(f'File {args.test_path} not found or error')
    is_gray = utils.is_image_gray(image)
    image = read_image(args.test_path, is_gray)
    print("{} image shape: {}".format("Gray" if is_gray else "RGB", image.shape))

    # Expand odd shape to even
    expend_W = False
    expend_H = False
    if image.shape[1] % 2 != 0:
        expend_W = True
        image = np.concatenate((image, image[:, -1, :][:, np.newaxis, :]), axis=1)
    if image.shape[2] % 2 != 0:
        expend_H = True
        image = np.concatenate((image, image[:, :, -1][:, :, np.newaxis]), axis=2)
    
    # Noise
    image = torch.FloatTensor([image]) # 1 * C(1 / 3) * W * H
    if args.add_noise:
        image = utils.add_batch_noise(image, args.noise_sigma)
    noise_sigma = torch.FloatTensor([args.noise_sigma])

    # Model & GPU
    model = FFDNet(is_gray=is_gray)
    #if args.cuda:
    if args.npu:
        #image = image.cuda()
        image = image.npu()
        #noise_sigma = noise_sigma.cuda()
        noise_sigma = noise_sigma.npu()
        #model = model.cuda()
        model = model.npu()

    # Dict
    model_path = args.model_path + ('net_gray.pth' if is_gray else 'net_rgb.pth')
    print(f"> Loading model param in {model_path}...")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    print('\n')
    
    # Test
    with torch.no_grad():
        start_time = time.time()
        image_pred = model(image, noise_sigma)
        stop_time = time.time()
        print("Test time: {0:.4f}s".format(stop_time - start_time))

    # PSNR
    psnr = utils.batch_psnr(img=image_pred, imclean=image, data_range=1)
    print("PSNR denoised {0:.2f}dB".format(psnr))

    # UnExpand odd
    if expend_W:
        image_pred = image_pred[:, :, :-1, :]
    if expend_H:
        image_pred = image_pred[:, :, :, :-1]

    # Save
    cv2.imwrite("ffdnet.png", utils.variable_to_cv2_image(image_pred))
    if args.add_noise:
        cv2.imwrite("noisy.png", utils.variable_to_cv2_image(image))

def main():
    parser = argparse.ArgumentParser()

    # Train
    parser.add_argument("--train_path", type=str, default='./train_data/',                  help='Train dataset dir.')
    parser.add_argument("--is_gray", action='store_true',                                   help='Train gray/rgb model.')
    parser.add_argument("--patch_size", type=int, default=32,                               help='Uniform size of training images patches.')
    parser.add_argument("--train_noise_interval", nargs=3, type=int, default=[0, 75, 15],   help='Train dataset noise sigma set interval.')
    parser.add_argument("--val_noise_interval", nargs=3, type=int, default=[0, 60, 30],     help='Validation dataset noise sigma set interval.')
    parser.add_argument("--batch_size", type=int, default=256,                              help='Batch size for training.')
    parser.add_argument("--epoches", type=int, default=80,                                  help='Total number of training epoches.')
    parser.add_argument("--val_epoch", type=int, default=5,                                 help='Total number of validation epoches.')
    parser.add_argument("--learning_rate", type=float, default=1e-3,                        help='The initial learning rate for Adam.')
    parser.add_argument("--save_checkpoints", type=int, default=5,                          help='Save checkpoint every epoch.')

    # Test
    parser.add_argument("--test_path", type=str, default='./test_data/color.png',           help='Test image path.')
    parser.add_argument("--noise_sigma", type=float, default=25,                            help='Input uniform noise sigma for test.')
    parser.add_argument('--add_noise', action='store_true',                                 help='Add noise_sigma to input or not.')

    # Global
    parser.add_argument("--model_path", type=str, default='./models/',                      help='Model loading and saving path.')
    parser.add_argument("--use_gpu", action='store_true',                                   help='Train and test using GPU.')
    parser.add_argument("--is_train", action='store_true',                                  help='Do train.')
    parser.add_argument("--is_test", action='store_true',                                   help='Do test.')
    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',                                      help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')
    parser.add_argument('--loss-scale-value', default=1024., type=float,                    help='loss scale using in amp, default -1 means dynamic')

    args = parser.parse_args()
    assert (args.is_train or args.is_test), 'is_train 鍜 is_test 鑷冲皯鏈変竴涓负 True'

    #args.cuda = args.use_gpu and torch.cuda.is_available()
    args.npu = args.use_gpu and torch.npu.is_available()
    print("> Parameters: ")
    for k, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print(f'\t{k}: {v}')
    print('\n')

    # Normalize noise level
    args.noise_sigma /= 255
    args.train_noise_interval[1] += 1
    args.val_noise_interval[1] += 1

    if args.is_train:
        train(args)

    if args.is_test:
        test(args)

if __name__ == "__main__":
    main()
