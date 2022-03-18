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
# Import Dependencies
import os
import copy
#import wandb
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils import AverageMeter, calculate_psnr
from model import ESPCN
from dataloader import get_data_loader
import time
import torch.npu
import os
import apex
from apex import amp
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def train(model, train_loader, device, criterion, optimizer):
    """ Function to train the model

    :param model: instance of model
    :param train_loader: training data loader
    :param device: training device, 'cpu', 'cuda'
    :param criterion: loss criterion, MSE
    :param optimizer: model optimizer, Adam
    :return: running training loss
    """

    model.train()
    running_loss = AverageMeter()
    step = 0
    for data in train_loader:
        start_time = time.time()
        inputs, labels = data

        inputs = inputs.to(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)
        labels = labels.to(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)

        prediction = model(inputs)
        loss = criterion(prediction, labels)
        running_loss.update(loss.detach(), len(inputs))

        optimizer.zero_grad()
        #loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        step_time = time.time() - start_time
        step += 1
        print("step:{}, Loss:{:.5f}, time/step(s):{:.5f}".format(step,loss.item(),step_time))
    return running_loss


def evaluate(model, val_loader, device, criterion):
    """ Function to evaluate the model

    :param model: instance of the model
    :param val_loader: validation data loader
    :param device: training device, 'cpu', 'cuda'
    :return: model predictions, running PSNR, running validation loss
    """

    # Evaluate the Model
    model.eval()
    running_psnr = AverageMeter()
    running_loss = AverageMeter()

    for data in val_loader:
        inputs, labels = data

        inputs = inputs.to(f'npu:{NPU_CALCULATE_DEVICE}')
        labels = labels.to(f'npu:{NPU_CALCULATE_DEVICE}')

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)
            loss = criterion(preds, labels)
            running_loss.update(loss.item(), len(inputs))

        running_psnr.update(calculate_psnr(preds, labels), len(inputs))

    print('eval psnr: {:.2f}'.format(running_psnr.avg))

    return preds, running_psnr, running_loss


def main(args):
    """ Main function to train/evaluate the model

    :param args: model input arguments
    :return: best trained model
    """

    # Set device
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
    cudnn.benchmark = True
    print(f"Device: {device}")

    # Set seed for reproducability
    torch.manual_seed(args.seed)

    # Setup wandb
    #wandb.init(project="ESPCN-PyTorch")
    #wandb.config.update(args)

    # Get dataloaders
    train_loader, val_loader = get_data_loader(dirpath_train=args.dirpath_train,
                                               dirpath_val=args.dirpath_val,
                                               scaling_factor=args.scaling_factor,
                                               patch_size=args.patch_size,
                                               stride=args.stride)

    for idx, (lr_image, hr_image) in enumerate(train_loader):
        print(f"Training - lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        print(f"Training - lr_image: {type(lr_image[0])}, hr_image: {type(hr_image[0])}")
        print(f"Training - lr_image: {lr_image[0]}, hr_image: {hr_image[0]}\n")
        break

    for idx, (lr_image, hr_image) in enumerate(val_loader):
        print(f"Validation - lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        print(f"Validation - lr_image: {type(lr_image[0])}, hr_image: {type(hr_image[0])}")
        print(f"Validation - lr_image: {lr_image[0]}, hr_image: {hr_image[0]}\n")
        break

    # Get the Model
    model = ESPCN(num_channels=1, scaling_factor=args.scaling_factor)
    model.to(f'npu:{NPU_CALCULATE_DEVICE}')

    #wandb.watch(model)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = apex.optimizers.NpuFusedAdam([
        {'params': model.feature_map_layer.parameters()},
        # As per paper, Sec 3.2, The final layer learns 10 times slower
        {'params': model.sub_pixel_layer.parameters(), 'lr': args.learning_rate * 0.1}
    ], lr=args.learning_rate)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2",loss_scale=128.0,combine_grad=True)
    # Train the Model
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in tqdm(range(args.epochs)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate * (0.1 ** (epoch // int(args.epochs * 0.8)))

        training_loss = train(model, train_loader, device, criterion, optimizer)
        torch.save(model.state_dict(), os.path.join(args.dirpath_out, 'epoch_{}.pth'.format(epoch)))

        preds, running_psnr, validation_loss = evaluate(model, val_loader, device, criterion)

        if running_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = running_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        print(f"\nEpoch: {epoch}, Training Loss: {training_loss.avg}, PSNR: {running_psnr.avg}, Validation Loss: {validation_loss.avg}\n")
        #wandb.log({"Epoch": epoch,
        #           "base_lr": args.learning_rate,
        #           "sub_pixel_layer_lr": param_group['lr'],
        #           "Training Loss": training_loss.avg,
        #           "PSNR": running_psnr.avg,
        #           "Validation Loss": validation_loss.avg})

        if epoch % args.log_interval == 0:
            img = preds[0].mul(255.0).cpu().squeeze().numpy()
            #wandb.log({"Upscaled Images": [wandb.Image(img, caption=f"PSNR: {running_psnr.avg}")]})

    print('Best Epoch: {}, PSNR: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.dirpath_out, 'best.pth'))

    return model.load_state_dict(best_weights)


def build_parser():
    parser = ArgumentParser(prog="ESPCN Dataset Preparation.")
    parser.add_argument("-t", "--dirpath_train", required=False, type=str,
                        help="Required. Path to training images dataset directory.")
    parser.add_argument("-v", "--dirpath_val", required=False, type=str,
                        help="Required. Path to validation images dataset directory.")
    parser.add_argument("-o", "--dirpath_out", required=False, type=str,
                        help="Required. Path to directory to save best model weights.")
    parser.add_argument("-ps", "--patch_size", default=17, required=False, type=int,
                        help="Optional. Sub-images patch size.")
    parser.add_argument("-sf", "--scaling_factor", default=3, required=False, type=int,
                        help="Optional. Image Up-scaling factor.")
    parser.add_argument("-s", "--stride", default=13, required=False, type=int,
                        help="Optional. Sub-image extraction stride.")
    parser.add_argument("-epochs", "--epochs", default=200, required=False, type=int,
                        help="Optional. Number of training epochs.")
    parser.add_argument("-lr", "--learning_rate", default=1e-3, required=False, type=float,
                        help="Optional. Learning Rate.")
    parser.add_argument("-log", "--log_interval", default=10, required=False, type=int,
                        help="Optional. Log model outputs after every n epochs.")
    parser.add_argument("-seed", "--seed", default=100, required=False, type=int,
                        help="Optional. Pytorch seed for reproducability.")

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    espcn_model = main(args)
