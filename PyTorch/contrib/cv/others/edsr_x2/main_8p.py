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
import os
import time
import shutil
import numpy as np
from apex import amp
from tqdm import tqdm

import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader

import utility
from model import edsr
from data import div2k
from option import args


def save_checkpoint(
    opt, epoch, checkpoint_performance, checkpoint_time, model, optimizer
):
    checkpoint_path = os.path.join(opt.save, "model_latest.pth")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "amp": amp.state_dict() if opt.amp else None,
        "best": np.array(checkpoint_performance)[:, 0].max(),
        "ck_performance": checkpoint_performance,
        "ck_time": checkpoint_time,
    }
    torch.save(checkpoint, checkpoint_path)
    if checkpoint_performance[-1][0] > opt.best:
        print(
            "=> This Epoch {} is the BEST: {}{}".format(
                epoch, checkpoint_performance[-1][0], opt.process_id
            )
        )
        opt.best = checkpoint_performance[-1][0]
        shutil.copyfile(
            checkpoint_path,
            os.path.join(opt.save, "model_best.pth"),
        )
    print("Best:", opt.best, "This Time:", checkpoint_performance[-1][0])


def test_eval(model, dataloader_test, opt):
    psnr_list = []

    for lr, hr, _ in tqdm(dataloader_test, ncols=80):
        with torch.no_grad():
            if opt.device == "npu":
                lr = lr.npu()
                hr = hr.npu()
            elif opt.device == "gpu":
                lr = lr.cuda()
                hr = hr.cuda()
            sr = model(lr)
        sr = utility.quantize(sr, opt.rgb_range)
        psnr_list.append(utility.calc_psnr(sr, hr, opt.scale, opt.rgb_range))
    return np.mean(psnr_list)


def prepare(opt):
    opt.process_device_map = utility.device_id_to_process_device_map(
        opt.device_list)
    utility.information_print(opt.process_device_map, mode=0)

    # Since we have ndevices_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    opt.ndevices_per_node = len(opt.process_device_map)
    opt.world_size = opt.ndevices_per_node * opt.world_size
    utility.information_print(
        "...multi processing...\nChoose to use {} {}s from device list...".format(
            opt.ndevices_per_node, opt.device
        ),
        mode=0,
    )

    if not os.path.exists(opt.save):
        os.makedirs(opt.save)
    else:
        utility.information_print(
            "The dir is existing, if continue, Retraining from and Replacing...",
            mode=0,
        )
    if opt.ifcontinue:
        if not os.path.exists(opt.checkpoint_path):
            raise ValueError(
                "Can't find this file for continuing train, please check"
            )
        print("Train from {}".format(opt.checkpoint_path))

    # To record the parameters
    with open(opt.save + "/Para.txt", "w") as f:
        for i in vars(opt):
            f.write(i + ":" + str(vars(opt)[i]) + "\n")
    f.close()


def main():
    print("===============main() start=================")
    opt = args
    opt.scale = 2
    utility.information_print(opt, mode=0)
    if opt.device == "npu":
        import torch.npu
    torch.manual_seed(opt.seed)
    prepare(opt)
    print("===============main() end=================")

    mp.spawn(main_worker, nprocs=opt.ndevices_per_node, args=[opt])


def main_worker(process_id, opt):
    # each sub_process choose device from process_device_map through process_id
    # process_id belongs to [0,...len(process_device_map)-1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29688"
    opt.sub_device_id = opt.process_device_map[process_id]
    if opt.device == "npu":
        torch.distributed.init_process_group(
            backend=opt.dist_backend, world_size=opt.world_size, rank=process_id
        )
    if opt.device == "gpu":
        torch.distributed.init_process_group(
            backend=opt.dist_backend,
            init_method="env://",
            world_size=opt.world_size,
            rank=process_id,
        )

    opt.process_id = process_id
    if opt.device == "npu":
        loc = "npu:{}".format(opt.sub_device_id)
        device = torch.device("npu:{}".format(opt.sub_device_id))
        torch.npu.set_device(device)
    elif opt.device == "gpu":
        loc = "cuda:{}".format(opt.sub_device_id)
        device = torch.device("cuda:{}".format(opt.sub_device_id))
        torch.cuda.set_device(opt.sub_device_id)

    print("====" * 5)
    print("SUB PROCESSING INFORMATION")
    print("Number of Mutil-process {}".format(opt.ndevices_per_node))
    print("rank ID {}".format(process_id))
    print("Wanted device {}ID:{}".format(opt.device, opt.sub_device_id))
    print(
        "Chosen device {} {}".format(
            opt.device,
            torch.cuda.current_device()
            if opt.device == "gpu"
            else torch.npu.current_device(),
        )
    )
    print("====" * 5)

    if not opt.test_only:
        criterion = nn.L1Loss()

    if opt.device == "npu":
        model = edsr.EDSR(opt).to(device)
        if not opt.test_only:
            from apex.optimizers import NpuFusedAdam
            optimizer = NpuFusedAdam(model.parameters(), lr=opt.lr)
    elif opt.device == "gpu":
        model = edsr.EDSR(opt).cuda()
        if not opt.test_only:
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if not opt.test_only:
        if opt.amp:
            if opt.device == "npu":
                model, optimizer = amp.initialize(
                    model,
                    optimizer,
                    opt_level=opt.opt_level,
                    loss_scale=opt.loss_scale,
                    combine_grad=True,
                )
            else:
                model, optimizer = amp.initialize(
                    model, optimizer, opt_level=opt.opt_level, loss_scale=opt.loss_scale
                )
    else:
        model = amp.initialize(
            model, opt_level=opt.opt_level, loss_scale=opt.loss_scale
        )

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], broadcast_buffers=False
    )

    if opt.test_only:
        if not os.path.isfile(opt.checkpoint_path):
            raise ValueError(
                "=> no checkpoint found at '{}'".format(opt.checkpoint_path)
            )
        print("loading checkpoint {}".format(opt.checkpoint_path))
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(opt.checkpoint_path, map_location=loc)
        model.load_state_dict(checkpoint["model"])
        checkpoint_performance = checkpoint["ck_performance"]
        checkpoint_time = checkpoint["ck_time"]
        if opt.amp:
            amp.load_state_dict(checkpoint["amp"])
        print(
            "loaded checkpoint '{}' (epoch {})".format(
                opt.checkpoint_path, checkpoint["epoch"]
            )
        )
        torch.backends.cudnn.benchmark = True

        # For testing model, every sub-process may carry out one time of testing the whole test dataloader
        dataset_test = div2k.DIV2K(opt, train=False)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)

        dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )

        model.eval()
        psnr_avg = test_eval(model, dataloader_test, opt)
        print("PSNR:", psnr_avg)
    else:
        if opt.ifcontinue:
            if not os.path.isfile(opt.checkpoint_path):
                raise ValueError(
                    "=> no checkpoint found at '{}'".format(
                        opt.checkpoint_path)
                )
            print("loading checkpoint: {}".format(opt.checkpoint_path))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(opt.checkpoint_path, map_location=loc)
            opt.start_epoch = checkpoint["epoch"]
            opt.best = checkpoint["best"]
            model.load_state_dict(checkpoint["model"])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            checkpoint_performance = checkpoint["ck_performance"]
            checkpoint_time = checkpoint["ck_time"]
            if opt.amp:
                amp.load_state_dict(checkpoint["amp"])
            print(
                "loaded checkpoint: {} (epoch {})".format(
                    opt.checkpoint_path, checkpoint["epoch"]
                )
            )
        else:
            opt.start_epoch = 0
            opt.best = -1
            checkpoint_performance = []
            checkpoint_time = []

        torch.backends.cudnn.benchmark = True
        dataset_train = div2k.DIV2K(opt)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train
        )
        opt.workers = int(
            (opt.workers + opt.ndevices_per_node - 1) / opt.ndevices_per_node
        )
        opt.batch_size = int(opt.batch_size / opt.ndevices_per_node)

        dataloader = DataLoader(
            dataset=dataset_train,
            batch_size=opt.batch_size,
            shuffle=(train_sampler is None),
            num_workers=opt.workers,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler,
        )

        # For testing model, every sub-process may carry out one time of testing the whole test dataloader
        dataset_test = div2k.DIV2K(opt, train=False)
        dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )
        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            # torch.cuda.synchronize()
            # To shufful between all epoch
            train_sampler.set_epoch(epoch)

            model.train()
            if opt.process_id % opt.ndevices_per_node == 0:
                epoch_losses = utility.AverageMeter()
                epoch_timer = utility.AverageMeter()
                timer_iter = utility.timer()
                with tqdm(
                    total=(
                        len(dataset_train)
                        - len(dataset_train) % (opt.batch_size *
                                                opt.ndevices_per_node)
                    ),
                    ncols=100,
                ) as _tqdm:
                    _tqdm.set_description(
                        "Epoch:{}/{}(RankID:{})".format(
                            epoch, opt.start_epoch + opt.epochs, opt.process_id
                        )
                    )
                    model.train()
                    for index, data in enumerate(dataloader):
                        if (
                            index > 3
                        ):  # the first three iterations don't be recorded in the time analysis
                            timer_iter.tic()  # start timer
                        inputs, labels, _ = data
                        if opt.device == "npu":
                            inputs, labels = inputs.to(
                                device), labels.to(device)
                        elif opt.device == "gpu":
                            inputs, labels = inputs.cuda(), labels.cuda()

                        preds = model(inputs)
                        loss = criterion(preds, labels)
                        optimizer.zero_grad()
                        if opt.amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        optimizer.step()
                        epoch_losses.update(loss.item(), len(inputs))

                        _tqdm.set_postfix(
                            loss="{:.6f}".format(epoch_losses.avg))
                        _tqdm.update(len(inputs) * opt.ndevices_per_node)
                        if opt.use_npu:
                            torch.npu.synchronize()
                        if (
                            index > 3
                        ):  # the first three inerations don't be recorded in the time analysis
                            timer_iter.hold()
                            epoch_timer.update(timer_iter.acc)
                    model.eval()
                    psnr_avg = test_eval(model, dataloader_test, opt)
                    checkpoint_time.append(epoch_timer.avg)
                    checkpoint_performance.append([psnr_avg])
                    save_checkpoint(
                        opt,
                        epoch,
                        checkpoint_performance,
                        checkpoint_time,
                        model,
                        optimizer,
                    )
                    print("Epoch:", epoch)
                    print("PSNR:", psnr_avg)
                    print("Loss:", epoch_losses.avg)
                    print("FPS:", 1000 * 8 / epoch_timer.avg)
            else:
                for index, data in enumerate(dataloader):
                    inputs, labels, _ = data
                    if opt.device == "npu":
                        inputs, labels = inputs.to(device), labels.to(device)
                    elif opt.device == "gpu":
                        inputs, labels = inputs.cuda(), labels.cuda()

                    preds = model(inputs)
                    loss = criterion(preds, labels)

                    optimizer.zero_grad()
                    if opt.amp:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()


if __name__ == "__main__":
    main()
