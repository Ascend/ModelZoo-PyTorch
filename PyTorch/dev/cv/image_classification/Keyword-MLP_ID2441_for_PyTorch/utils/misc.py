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
"""Miscellaneous helper functions."""

import torch
from torch import nn, optim
import numpy as np
import random
import os
#import wandb
from models.kwmlp import KW_MLP
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def seed_everything(seed: str) -> None:
    """Set manual seed.

    Args:
        seed (int): Supplied seed.
    """
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'Set seed {seed}')


def count_params(model: nn.Module) -> int:
    """Counts number of parameters in a model.

    Args:
        model (torch.nn.Module): Model instance for which number of params is to be counted.

    Returns:
        int: Parameter count.
    """
    
    return sum(map(lambda p: p.data.numel(), model.parameters()))


def calc_step(epoch: int, n_batches: int, batch_index: int) -> int:
    """Calculates current step.

    Args:
        epoch (int): Current epoch.
        n_batches (int): Number of batches in dataloader.
        batch_index (int): Current batch index.

    Returns:
        int: Current step.
    """
    return (epoch - 1) * n_batches + (1 + batch_index)


def log(log_dict: dict, step: int, config: dict) -> None:
    """Handles logging for metric tracking server, local disk and stdout.

    Args:
        log_dict (dict): Log metric dict.
        step (int): Current step.
        config (dict): Config dict.
    """

    # send logs to wandb tracking server
    if config["exp"]["wandb"]:
        wandb.log(log_dict, step=step)

    log_message = f"Step: {step} | " + " | ".join([f"{k}: {v}" for k, v in log_dict.items()])

    # write logs to disk
    if config["exp"]["log_to_file"]:
        log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")
    
        with open(log_file, "a+") as f:
            f.write(log_message + "\n")

    # show logs in stdout
    if config["exp"]["log_to_stdout"]:
        print(log_message)


def get_model(model_config: dict) -> nn.Module:
    """Creates model from config dict.

    Args:
        model_config (dict): Dict containing model config params. If the "name" key is not None, other params are ignored.

    Returns:
        nn.Module: Model instance.
    """
    
    if model_config["type"] == "kw-mlp":
        return KW_MLP(**model_config)
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")


def save_model(epoch: int, val_acc: float, save_path: str, net: nn.Module, optimizer : optim.Optimizer = None, scheduler: optim.lr_scheduler._LRScheduler = None, log_file : str = None) -> None:
    """Saves checkpoint.

    Args:
        epoch (int): Current epoch.
        val_acc (float): Validation accuracy.
        save_path (str): Checkpoint path.
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer, optional): Optimizer. Defaults to None.
        scheduler (optim.lr_scheduler._LRScheduler): Scheduler. Defaults to None.
        log_file (str, optional): Log file. Defaults to None.
    """

    ckpt_dict = {
        "epoch": epoch,
        "val_acc": val_acc,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else optimizer,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else scheduler
    }

    torch.save(ckpt_dict, save_path)

    log_message = f"Saved {save_path} with accuracy {val_acc}."
    print(log_message)
    
    if log_file is not None:
        with open(log_file, "a+") as f:
            f.write(log_message + "\n")
    
