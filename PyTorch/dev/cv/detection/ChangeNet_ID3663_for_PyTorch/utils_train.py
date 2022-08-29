# MIT License
#
# Copyright (c) 2020 xxx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================
#
# Copyright 2021 Huawei Technologies Co., Ltd
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import seg_metrics
import copy
import apex
import time
import os

def validation(model, val_loader, criterion, epoch, writer, device):
    model.eval()

    running_loss = 0.0
    # running_corrects = 0
    list_dice_val = []

    # Iterate over data.
    for sample in val_loader:
        reference_img = sample['reference'].to(device, non_blocking=True)
        test_img = sample['test'].to(device, non_blocking=True)
        labels = (sample['label'] > 0).squeeze(1).type(torch.LongTensor).to(device, non_blocking=True)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(reference_img, test_img)

            # Calculate Loss
            loss = criterion(outputs, labels)

            # Get the correct class by looking for the max value across channels
            _, preds = torch.max(outputs, 1)

            # Calculate metric during evaluation
            dice_value = seg_metrics.iou_segmentation(preds.type(torch.LongTensor),
                                                      labels.type(torch.LongTensor))
            list_dice_val.append(dice_value.item())

        # statistics
        running_loss += loss.item()
        # running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader)
    # epoch_acc = running_corrects.double() / len(val_loader.dataset)
    epoch_acc = np.mean(list_dice_val)

    print('val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    writer.add_scalar('epoch/loss_val', epoch_loss, epoch)
    writer.add_scalar('metrics/iou_val', epoch_acc, epoch)

    return epoch_acc

def train_model(model, dataloaders, criterion, optimizer, sc_plt, writer, device, args):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    iterations = 0

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    for epoch in range(args.num_epochs):
        if args.local_rank in [0, -1]:
            print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
            print('-' * 10)
        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        running_loss = 0.0
        # running_corrects = 0

        if args.local_rank != -1:
            train_loader.sampler.set_epoch(epoch)

        # Iterate over data.
        # outputs: torch.Size([20, 11, 224, 224])
        # labels: torch.Size([20, 1, 224, 224])
        for sample in train_loader:
            start_time = time.perf_counter()
            reference_img = sample['reference'].to(device, non_blocking=True)
            test_img = sample['test'].to(device, non_blocking=True)
            labels = (sample['label'] > 0).squeeze(1).type(torch.LongTensor).to(device, non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model(reference_img, test_img)

                # Calculate Loss
                loss = criterion(outputs, labels)

                # Get the correct class by looking for the max value across channels
                # preds: torch.Size([20, 224, 224])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if args.fp16:
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()

            end_time = time.perf_counter()
            # statistics
            running_loss += loss.item()
            # running_corrects += torch.sum(preds == labels.data)
            iterations += 1
            if args.local_rank in [0, -1] and iterations == len(train_loader) * (epoch+1) -1:
                step_time = end_time - start_time
                FPS = reference_img.size(0) / step_time
                print("{} FPS: {:.4f}".format(iterations, FPS))
                # Calculate 1/10th of batch size
                num_imgs = reference_img.shape[0] // 10
                writer.add_images('/run/preds', preds[0:num_imgs].unsqueeze(1), iterations)
                writer.add_images('/run/labels', labels[0:num_imgs].unsqueeze(1), iterations)

        epoch_loss = running_loss / len(train_loader)
        # epoch_acc = running_corrects.double() / len(train_loader)

        # Update Scheduler if training loss doesn't change for patience(2) epochs
        sc_plt.step(epoch_loss)
        if args.local_rank in [0, -1]:
            print('train Loss: {:.4f}'.format(epoch_loss))
            writer.add_scalar('epoch/loss_train', epoch_loss, epoch)
            # Get current learning rate (To display on Tensorboard)
            for param_group in optimizer.param_groups:
                curr_learning_rate = param_group['lr']
                writer.add_scalar('epoch/learning_rate_train', curr_learning_rate, epoch)

            val_epoch_acc = validation(model, val_loader, criterion, epoch, writer, device)
            # deep copy the model and save if accuracy is better
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    if args.local_rank in [0, -1]:
        print('Best val Acc: {:4f}'.format(best_acc))
        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pkl'))