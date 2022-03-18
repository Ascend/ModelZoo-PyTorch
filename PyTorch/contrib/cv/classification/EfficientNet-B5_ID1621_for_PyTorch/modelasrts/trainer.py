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

from pycls.core.net import unwrap_model
import glob
from apex import amp 
from pycls.core.config import cfg
from pycls.core.cuda import ApexScaler
import os
import shutil

import moxing as mox

import torch.onnx
from collections import OrderedDict
from pycls.models.effnet import EffNet
from iopath.common.file_io import g_pathmgr
import pycls.core.config as config

logger = logging.get_logger(__name__)

cur_step = 0
CACHE_TRAINING_URL = "/cache/training/"
CACHE_DATA_URL = "/cache/data_url"
CACHE_MODEL_URL = "/cache/model"
 
def setup_model():
    
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_model()
    logger.info("Model:\n{}".format(model)) if cfg.VERBOSE else ()
    # Log model complexity
    logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    #assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    assert cfg.NUM_GPUS <= torch.npu.device_count(), err_str   
    cur_device = torch.npu.current_device()
    model = model.to(cur_device)
    optimizer = optim.construct_optimizer(model)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128)
    if cfg.NUM_GPUS > 1:
        #Make model replica operate on the current device
        ddp = torch.nn.parallel.DistributedDataParallel
        model = ddp(model, device_ids=[cur_device],  broadcast_buffers=False)
    
    return model,optimizer
    

def train_epoch(loader, model, loss_fun, optimizer, scaler, meter, cur_epoch):
    """Performs one epoch of training."""
    # Shuffle the data
    data_loader.shuffle(loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs = inputs.npu()
        labels = labels.to(torch.int32).npu()    
        labels = labels.to(non_blocking=False)
        # Convert labels to smoothed one-hot vector
        p_labels = labels[:]
        labels_one_hot = net.smooth_one_hot_labels(labels).npu()   
        # Apply mixup to the batch (no effect if mixup alpha is 0)
        inputs, labels_one_hot, labels = net.mixup(inputs, labels_one_hot)
        # Perform the forward pass and compute the loss
        preds = model(inputs)
        loss = loss_fun(preds, labels_one_hot)
        # Perform the backward pass and update the parameters
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()        
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
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)

@torch.no_grad()
def test_epoch(loader, model, meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs = inputs.npu()
        labels = labels.to(torch.int32).npu()     
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
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)

def train_model(args):
    """Trains the model."""
    # Setup training/testing environment
    env.setup_env()
    # Construct the model, loss_fun, and optimizer
    model,optimizer = setup_model()
    loss_fun = builders.build_loss_fun().npu()
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
        #cp.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        CACHE_MODEL_URL = "/cache/model"
        os.makedirs(CACHE_MODEL_URL, exist_ok=True)
        mox.file.copy_parallel(cfg.TRAIN.WEIGHTS, os.path.join(CACHE_MODEL_URL, "model.pyth"))
        pretrained_weight = os.path.join(CACHE_MODEL_URL, "model.pyth")
        cp.load_checkpoint(pretrained_weight, model)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = data_loader.construct_train_loader(args)
    test_loader = data_loader.construct_test_loader(args)
    # modelarts modification
    '''
    real_path = '/cache/data_url'
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    mox.file.copy_parallel(args.data_url, real_path)
    print("training data finish copy to %s." % real_path)
    train_loader = os.path.join(args.data, 'train')
    test_loader = os.path.join(real_path, 'val')
    '''
    # modelarts modification
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    # Create a GradScaler for mixed precision training
    scaler = ApexScaler()    
    # Compute model and loader timings
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
         benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    best_err = np.inf
    
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        params = (train_loader, model, loss_fun, optimizer, scaler, train_meter)
        train_epoch(*params, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        # Evaluate the model
        test_epoch(test_loader, model, test_meter, cur_epoch)
        # Check if checkpoint is best so far (note: should checkpoint meters as well)
        stats = test_meter.get_epoch_stats(cur_epoch)
        best = stats["top1_err"] <= best_err
        best_err = min(stats["top1_err"], best_err)
        # Save a checkpoint
        file = cp.save_checkpoint(model, optimizer, cur_epoch, best)
        logger.info("Wrote checkpoint to: {}".format(file))
    
    # --------------modelarts modification----------    
    if args.onnx:
        convert_pth_to_onnx(args)
    mox.file.copy_parallel(CACHE_TRAINING_URL, args.train_url)
    # --------------modelarts modification end----------
    
def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    env.setup_env()
    # Construct the model
    model,optimizer = setup_model()    
    # Load model weights
    cp.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = data_loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)


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

def convert_pth_to_onnx(args):
    os.makedirs(CACHE_TRAINING_URL, exist_ok=True)
    shutil.copy(os.path.join(cfg.OUT_DIR, 'model.pyth'), CACHE_TRAINING_URL)
    pth_pattern = os.path.join(CACHE_TRAINING_URL, 'model.pyth')
    pth_file_list = glob.glob(pth_pattern)
    if not pth_file_list:
        print("can't find pth {pth_pattern}")
    pth_file = pth_file_list[0]
    onnx_path = pth_file.split(".")[0] + '.onnx'
    convert(pth_file, onnx_path)

def proc_node_module(checkpoint, attr_name):
    new_model_state = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0: 7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_model_state[name] = v
    return new_model_state
    
def convert(pth_file_path, onnx_file_path):
    """Sets up a model for training or testing and log the results."""
    loc = 'cpu'
    with g_pathmgr.open(pth_file_path, "rb") as f:
        checkpoint = torch.load(f, map_location=loc)       
    cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../configs/dds_baselines/effnet/EN-B5_dds_1npu_full.yaml")
    config.merge_from_file(cfg_file)
    cfg.freeze()
    model = EffNet().to(loc)
    checkpoint['model_state'] = proc_node_module(checkpoint, 'model_state')
    model.load_state_dict(checkpoint["model_state"], False)   
    model = model.to(loc)
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 3, 456, 456).to(loc)
    torch.onnx.export(model, dummy_input, onnx_file_path, input_names = input_names, output_names = output_names, opset_version=11)