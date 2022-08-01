#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Tools for training and testing a model."""
from copy import deepcopy

import numpy as np
import pycls.core.benchmark as benchmark
import pycls.core.builders as builders
import pycls.core.checkpoint as cp
import pycls.core.distributed as dist
import pycls.core.env as env
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.datasets.loader as data_loader
import torch
import torch.npu

from apex import amp 
from pycls.core.config import cfg

logger = logging.get_logger(__name__)


def setup_model(device_type="npu"):

    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_model()
    logger.info("Model:\n{}".format(model)) if cfg.VERBOSE else ()
    # Log model complexity
    logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    if device_type == "npu":
        err_str = "Cannot use more NPU devices than available"
        #assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
        assert cfg.NUM_GPUS <= torch.npu.device_count(), err_str   
        cur_device = torch.npu.current_device()
        loc = "npu:{}".format(cur_device)
        model = model.to(loc)
    elif device_type == "gpu":
        err_str = "Cannot use more GPU devices than available"
        #assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
        assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str   
        cur_device = torch.cuda.current_device()
        model = model.to(cur_device)
    ema = deepcopy(model)
    optimizer = optim.construct_optimizer(model)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale='dynamic')
    if cfg.NUM_GPUS > 1:
        #Make model replica operate on the current device
        ddp = torch.nn.parallel.DistributedDataParallel
        model = ddp(model, device_ids=[cur_device], broadcast_buffers=False)
    return model, ema, optimizer
    

def profiling(loader, model, loss_fun, optimizer, device_type):
    # switch to train mode
    model.train()

    def update(model, inputs, labels, optimizer):
        labels_one_hot = net.smooth_one_hot_labels(labels).npu()   
        # Apply mixup to the batch (no effect if mixup alpha is 0)
        inputs, labels_one_hot, labels = net.mixup(inputs, labels_one_hot)
        preds = model(inputs)
        loss = loss_fun(preds, labels_one_hot)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

    for cur_iter, (inputs, labels) in enumerate(loader):
        if device_type == "npu":
            inputs = inputs.npu()
            labels = labels.to(torch.int32).npu()
            labels = labels.to(non_blocking=False)
        elif device_type == "gpu":
            inputs = inputs.cuda()
            labels = labels.to(torch.int32).cuda()
            labels = labels.to(non_blocking=False)
            
        if cur_iter < 5:
            update(model, inputs, labels, optimizer)
        else:
            if device_type == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(model, inputs, labels, optimizer)
                prof.export_chrome_trace("EfficientNet-B1_NPU1P.prof")
            elif device_type == "gpu":
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(model, inputs, labels, optimizer)
                prof.export_chrome_trace("EfficientNet-B1_GPU1P.prof")
            break

    		


def train_epoch(loader, model, ema, loss_fun, optimizer, meter, device_type, cur_epoch):
    """Performs one epoch of training."""
    # Shuffle the data
    data_loader.shuffle(loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    ema.train()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(loader):
        if device_type == "npu":

            # Transfer the data to the current GPU device
            inputs = inputs.npu()
            labels = labels.to(torch.int32).npu()
            labels = labels.to(non_blocking=False)
            p_labels = labels[:]
            # Convert labels to smoothed one-hot vector
            labels_one_hot = net.smooth_one_hot_labels(labels).npu()   
            # Apply mixup to the batch (no effect if mixup alpha is 0)
            inputs, labels_one_hot, labels = net.mixup(inputs, labels_one_hot)
            # Perform the forward pass and compute the loss
            preds = model(inputs)
            loss = loss_fun(preds, labels_one_hot)
            optimizer.zero_grad()
            stream = torch.npu.current_stream()
            stream.synchronize()
            # Perform the backward pass and update the parameters
            stream = torch.npu.current_stream()
            stream.synchronize()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            stream = torch.npu.current_stream()
            stream.synchronize()
            optimizer.step()
            net.update_model_ema(model, ema, cur_epoch, cur_iter)
            stream = torch.npu.current_stream()
            stream.synchronize()
        elif device_type == "gpu":
            # Transfer the data to the current GPU device
            inputs = inputs.cuda()
            labels = labels.to(torch.int32).cuda()    
            labels = labels.to(non_blocking=False)
            # Convert labels to smoothed one-hot vector
            labels_one_hot = net.smooth_one_hot_labels(labels).cuda()   
            # Apply mixup to the batch (no effect if mixup alpha is 0)
            inputs, labels_one_hot, labels = net.mixup(inputs, labels_one_hot)
            # Perform the forward pass and compute the loss
            preds = model(inputs)
            loss = loss_fun(preds, labels_one_hot)
            stream = torch.cuda.current_stream()
            stream.synchronize()
            # Perform the backward pass and update the parameters
            optimizer.zero_grad()
            stream = torch.cuda.current_stream()
            stream.synchronize()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            stream = torch.cuda.current_stream()
            stream.synchronize()
            optimizer.step()
            net.update_model_ema(model, ema, cur_epoch, cur_iter)
            stream = torch.cuda.current_stream()
            stream.synchronize()              
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, p_labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        # loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        meter.iter_toc()
        # Update and log stats
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
        if cur_iter == 4:
            meter.reset()
            meter.iter_tic()              
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)

@torch.no_grad()
def test_epoch(loader, model, meter, device_type, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        if device_type == "npu":
            inputs = inputs.npu()
            labels = labels.to(torch.int32).npu()     
            labels = labels.to(non_blocking=False)
        elif device_type == "gpu":
            inputs = inputs.cuda()
            labels = labels.to(torch.int32).cuda()     
            labels = labels.to(non_blocking=False)            
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        meter.iter_toc()
        # Update and log stats
        meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
        if cur_iter == 4:
            meter.reset()
            meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)


def train_model(device_type="npu", is_profile=0):
    """Trains the model."""
    # Setup training/testing environment
    env.setup_env()
    # Construct the model, loss_fun, and optimizer
    model, ema, optimizer = setup_model(device_type)
    if device_type == "npu":
        loss_fun = builders.build_loss_fun().npu()
    elif device_type == "gpu":
        loss_fun = builders.build_loss_fun().cuda()
    # Load checkpoint or initial weights
    start_epoch = 0

    if cfg.TRAIN.PRETRAINED:
        cp.pretrained_load_checkpoint(cfg.TRAIN.PRETRAINED, model)
        logger.info("Loaded pretrained initial weights from: {}".format(cfg.TRAIN.PRETRAINED))
    elif cfg.TRAIN.AUTO_RESUME and cp.has_checkpoint():
        file = cp.get_last_checkpoint()
        epoch = cp.load_checkpoint(file, model, optimizer)
        logger.info("Loaded checkpoint from: {}".format(file))
        start_epoch = epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        cp.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    ema_meter = meters.TestMeter(len(test_loader), "test_ema")
    # Compute model and loader timings
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
         benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    best_err = np.inf
    if is_profile and cfg.NUM_GPUS == 1:
        logger.info("========Do profile========")
        profiling(train_loader, model, loss_fun, optimizer, device_type)
        logger.info("========Done profile========")
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        params = (train_loader, model, ema, loss_fun, optimizer, train_meter, device_type)
        train_epoch(*params, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
            net.compute_precise_bn_stats(ema, train_loader)
        # Evaluate the model
        test_epoch(test_loader, model, test_meter, device_type, cur_epoch)
        test_epoch(test_loader, ema, ema_meter, device_type, cur_epoch)
        # Check if checkpoint is best so far (note: should checkpoint meters as well)
        stats = test_meter.get_epoch_stats(cur_epoch)
        ema_stats =  ema_meter.get_epoch_stats(cur_epoch)
        best = min(ema_stats["top1_err"],stats["top1_err"]) <= best_err
        best_err = min(min(ema_stats["top1_err"],stats["top1_err"]), best_err)
        # Save a checkpoint
        if ema_stats["top1_err"] < stats["top1_err"]:
            file = cp.save_checkpoint(ema, optimizer, cur_epoch, best)
        else:
            file = cp.save_checkpoint(model, optimizer, cur_epoch, best)
        logger.info("Wrote checkpoint to: {}".format(file))

   
def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    env.setup_env()
    # Construct the model
    model, ema, optimizer = setup_model()
    # Load model weights
    cp.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = data_loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, "npu", 0)


def time_model():
    """Times model."""
    # Setup training/testing environment
    env.setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().npu()
    # Compute model and loader timings
    benchmark.compute_time_model(model, loss_fun)


def time_model_and_loader():
    """Times model and data loader."""
    # Setup training/testing environment
    env.setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().npu()
    # Create data loaders
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    # Compute model and loader timings
    benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
