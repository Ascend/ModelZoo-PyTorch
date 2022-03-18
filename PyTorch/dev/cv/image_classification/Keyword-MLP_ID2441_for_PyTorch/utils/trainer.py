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
import torch
from torch import nn, optim
from typing import Callable, Tuple
from torch.utils.data import DataLoader
from utils.misc import log, save_model
import os
import time
from tqdm import tqdm
import torch.npu
import os
try:
    import apex
    from apex import amp
except:
    amp = None
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def train_single_batch(net: nn.Module, data: torch.Tensor, targets: torch.Tensor, optimizer: optim.Optimizer, criterion: Callable, device: torch.device) -> Tuple[float, int]:
    """Performs a single training step.

    Args:
        net (nn.Module): Model instance.
        data (torch.Tensor): Data tensor, of shape (batch_size, dim_1, ... , dim_N).
        targets (torch.Tensor): Target tensor, of shape (batch_size).
        optimizer (optim.Optimizer): Optimizer instance.
        criterion (Callable): Loss function.
        device (torch.device): Device.

    Returns:
        float: Loss scalar.
        int: Number of correct preds.
    """

    data, targets = data.to(f'npu:{NPU_CALCULATE_DEVICE}'), targets.to(f'npu:{NPU_CALCULATE_DEVICE}')

    optimizer.zero_grad()
    outputs = net(data)
    loss = criterion(outputs, targets)
    with amp.scale_loss(loss,optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()

    correct = outputs.argmax(1).eq(targets).sum()
    return loss.item(), correct.item()


@torch.no_grad()
def evaluate(net: nn.Module, criterion: Callable, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Performs inference.

    Args:
        net (nn.Module): Model instance.
        criterion (Callable): Loss function.
        dataloader (DataLoader): Test or validation loader.
        device (torch.device): Device.

    Returns:
        accuracy (float): Accuracy.
        float: Loss scalar.
    """

    net.eval()
    correct = 0
    running_loss = 0.0

    for data, targets in tqdm(dataloader):
        data, targets = data.to(f'npu:{NPU_CALCULATE_DEVICE}'), targets.to(f'npu:{NPU_CALCULATE_DEVICE}')
        out = net(data)
        correct += out.argmax(1).eq(targets).sum().item()
        loss = criterion(out, targets)
        running_loss += loss.item()

    net.train()
    accuracy = correct / len(dataloader.dataset)
    return accuracy, running_loss / len(dataloader)


def train(net: nn.Module, optimizer: optim.Optimizer, criterion: Callable, trainloader: DataLoader, valloader: DataLoader, schedulers: dict, config: dict) -> None:
    """Trains model.

    Args:
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer): Optimizer instance.
        criterion (Callable): Loss function.
        trainloader (DataLoader): Training data loader.
        valloader (DataLoader): Validation data loader.
        schedulers (dict): Dict containing schedulers.
        config (dict): Config dict.
    """
    
    step = 0
    best_acc = 0.0
    n_batches = len(trainloader)
    device = config["hparams"]["device"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")
    
    ############################
    # start training
    ############################
    net.train()
    
    for epoch in range(config["hparams"]["start_epoch"], config["hparams"]["n_epochs"]):
        t0 = time.time()
        running_loss = 0.0
        correct = 0

        for batch_index, (data, targets) in enumerate(trainloader):
            start_time = time.time()

            if schedulers["warmup"] is not None and epoch < config["hparams"]["scheduler"]["n_warmup"]:
                schedulers["warmup"].step()

            elif schedulers["scheduler"] is not None:
                schedulers["scheduler"].step()

            ####################
            # optimization step
            ####################

            loss, corr = train_single_batch(net, data, targets, optimizer, criterion, device)
            running_loss += loss
            correct += corr

            if not step % config["exp"]["log_freq"]:       
                log_dict = {"epoch": epoch, "loss": loss, "lr": optimizer.param_groups[0]["lr"]}
                log(log_dict, step, config)

            step += 1
            step_time = time.time() - start_time
            FPS = config["hparams"]["batch_size"] / step_time
            print("Epoch:{}, step:{}, time/step(s):{:.4f}, FPS:{:.3f}".format(epoch,step,step_time,FPS))
        #######################
        # epoch complete
        #######################

        log_dict = {"epoch": epoch, "time_per_epoch": time.time() - t0, "train_acc": correct/(len(trainloader.dataset)), "avg_loss_per_ep": running_loss/len(trainloader)}
        log(log_dict, step, config)

        if not epoch % config["exp"]["val_freq"]:
            val_acc, avg_val_loss = evaluate(net, criterion, valloader, device)
            log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
            log(log_dict, step, config)

            # save best val ckpt
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(config["exp"]["save_dir"], "best.pth")
                save_model(epoch, val_acc, save_path, net, optimizer, schedulers["scheduler"], log_file) 

    ###########################
    # training complete
    ###########################

    val_acc, avg_val_loss = evaluate(net, criterion, valloader, device)
    log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
    log(log_dict, step, config)

    # save final ckpt
    save_path = os.path.join(config["exp"]["save_dir"], "last.pth")
    save_model(epoch, val_acc, save_path, net, optimizer, schedulers["scheduler"], log_file)
