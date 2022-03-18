# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

import os
import time
import socket
import logging

import torch

from . import metric
from . import callback

import pdb
from apex import amp

"""
Static Model
"""
class static_model(object):

    def __init__(self,
                 net,
                 criterion=None,
                 model_prefix='',
                 single_checkpoint=False,
                 **kwargs):
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        # init params
        self.net = net
        self.model_prefix = model_prefix
        self.criterion = criterion
        self.single_checkpoint = single_checkpoint
        if self.single_checkpoint and torch.distributed.is_initialized:
            logging.warning(">> only keeping the checkpoint for rank=0 node, making sure you are using shared filesystem")

        self.is_distributed = False
        self.net_without_ddp = net

        print("======================not self.single_checkpoint====================")
        print(not self.single_checkpoint)

    def set_net_without_ddp(self, net_without_ddp):
        self.net_without_ddp = net_without_ddp

    def set_distributed_mode(self):
        self.is_distributed = True

    def load_state(self, state_dict, strict=False):
        logging.info(f"==================load param (rank:{torch.distributed.get_rank() if self.is_distributed else 0})==================")
        if strict:
            logging.info("==================strictly load_state...==================")
            self.net_without_ddp.load_state_dict(state_dict=state_dict)
        else:
            logging.info("==================foreach name & param copy (customized partialy load function)...==================")
            # customized partialy load function
            net_state_keys = list(self.net_without_ddp.state_dict().keys())
            for name, param in state_dict.items():
                if name in self.net_without_ddp.state_dict().keys():
                    dst_param_shape = self.net_without_ddp.state_dict()[name].shape
                    if param.shape == dst_param_shape:
                        self.net_without_ddp.state_dict()[name].copy_(param.view(dst_param_shape))
                        net_state_keys.remove(name)
            # indicating missed keys
            if net_state_keys:
                logging.warning(">> Failed to load: {}".format(net_state_keys))
                return False
        return True

    def get_checkpoint_path(self, epoch, suffix=''):
        assert self.model_prefix, "model_prefix undefined!"
        if torch.distributed.is_initialized and not self.single_checkpoint:
            # pth_mark = socket.gethostname()
            pth_mark = str(torch.distributed.get_rank())
            checkpoint_path = "{}_rank-{}_ep-{:04d}{}.pth".format(self.model_prefix, pth_mark, epoch, suffix)
        else:
            checkpoint_path = "{}_ep-{:04d}{}.pth".format(self.model_prefix, epoch, suffix)
        return checkpoint_path

    def load_checkpoint(self, epoch, optimizer=None, suffix=''):

        load_path = self.get_checkpoint_path(epoch, suffix)
        assert os.path.exists(load_path), "Failed to load: {} (file not exist)".format(load_path)

        checkpoint = torch.load(load_path)

        all_params_matched = self.load_state(checkpoint['state_dict'], strict=False)

        if optimizer:
            if 'optimizer' in checkpoint.keys() and all_params_matched:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info("Model & Optimizer states are resumed from: `{}'".format(load_path))
            else:
                logging.warning(">> Failed to load optimizer state from: `{}'".format(load_path))
        else:
            logging.info("Only model state resumed from: `{}'".format(load_path))

        if 'epoch' in checkpoint.keys():
            if checkpoint['epoch'] != epoch:
                logging.warning(">> Epoch information inconsistant: {} vs {}".format(checkpoint['epoch'], epoch))

    def save_checkpoint(self, epoch, optimizer_state=None, suffix=''):

        if self.is_distributed and self.single_checkpoint and torch.distributed.is_initialized and torch.distributed.get_rank() != 0:
            logging.info("Checkpoint saved by node 0 (rank=0). Now is rank: {}".format(torch.distributed.get_rank()))
            return

        save_path = self.get_checkpoint_path(epoch, suffix)
        save_folder = os.path.dirname(save_path)
        logging.info(f"===================save folder (rank:{torch.distributed.get_rank() if self.is_distributed else 0})===================")
        logging.info(f"save_folder:{save_folder}, save_path:{save_path}")

        if not os.path.exists(save_folder):
            logging.debug("mkdir {}".format(save_folder))
            os.makedirs(save_folder)

        if not optimizer_state:
            torch.save({'epoch': epoch,
                        'state_dict': self.net_without_ddp.state_dict()},
                        save_path)
            logging.info("Checkpoint (only model) saved to: {}".format(save_path))
        else:
            torch.save({'epoch': epoch,
                        'state_dict': self.net_without_ddp.state_dict(),
                        'optimizer': optimizer_state},
                        save_path)
            logging.info("Checkpoint (model & optimizer) saved to: {}".format(save_path))


    def forward(self, data, target=None):
        """ typical forward function with:
            single output and single loss
        """
        input_var = data.float().npu()
        target_var = target.npu()

        output = self.net(input_var)
        if hasattr(self, 'criterion') and self.criterion is not None \
            and target is not None:
            loss = self.criterion(output, target_var)
        else:
            loss = None
        return [output], [loss]


"""
Typical model that able to update itself
"""
class model(static_model):

    """
    parameter for initializa fit
    """
    def __init__(self,
                 net,
                 criterion,
                 model_prefix='',
                 step_callback=None,
                 step_callback_freq=20,
                 epoch_callback=None,
                 save_checkpoint_freq=1,
                 opt_batch_size=None,
                 distributed=False,
                 args=None,
                 **kwargs):

        # load parameters
        super(model, self).__init__(net, criterion=criterion,
                                         model_prefix=model_prefix,
                                         **kwargs)

        # load optional arguments
        # - callbacks
        self.callback_kwargs = {'lr': None,
                                'epoch': None,
                                'batch': None,
                                'batch_elapse': None,
                                'update_elapse': None,
                                'epoch_elapse': None,
                                'namevals': None,
                                'optimizer_dict': None,}

        if not step_callback:
            step_callback = callback.CallbackList(callback.SpeedMonitor(),
                                                  callback.MetricPrinter())
        if not epoch_callback:
            epoch_callback = (lambda **kwargs: None)

        self.step_callback = step_callback
        self.step_callback_freq = step_callback_freq
        self.epoch_callback = epoch_callback
        self.save_checkpoint_freq = save_checkpoint_freq
        self.batch_size=opt_batch_size

        self.distributed = distributed
        self.args = args

    """
    Inorder to customize the callback function,
    you will have to overwrite the functions below
    """
    def step_end_callback(self):
        if self.distributed and torch.distributed.is_initialized and torch.distributed.get_rank() != 0:
            return

        self.step_callback(**(self.callback_kwargs))

    def epoch_end_callback(self):
        if self.distributed and torch.distributed.is_initialized and torch.distributed.get_rank() != 0:
            return

        self.epoch_callback(**(self.callback_kwargs))
        if self.callback_kwargs['epoch_elapse'] is not None:
            logging.info("Epoch [{:d}]   time cost: {:.2f} sec ({:.2f} h)".format(
                    self.callback_kwargs['epoch'],
                    self.callback_kwargs['epoch_elapse'],
                    self.callback_kwargs['epoch_elapse']/3600.))
        if self.callback_kwargs['epoch'] == 0 \
           or ((self.callback_kwargs['epoch']+1) % self.save_checkpoint_freq) == 0:
            self.save_checkpoint(epoch=self.callback_kwargs['epoch']+1,
                                 optimizer_state=self.callback_kwargs['optimizer_dict'])

    """
    Optimization
    """
    def adjust_learning_rate(self, lr, optimizer):
        for param_group in optimizer.param_groups:
            if 'lr_mult' in param_group:
                lr_mult = param_group['lr_mult']
            else:
                lr_mult = 1.0
            param_group['lr'] = lr * lr_mult
        return lr


    """
    parameters for deploy training, e.g. device, dataset
    """
    def fit(self, train_iter, optimizer, lr_scheduler,
            eval_iter=None,
            metrics=metric.Accuracy(topk=1),
            epoch_start=0,
            epoch_end=10000,
            precise_bn=False,
            precise_bn_steps=500,
            epoch_div_factor=1,
            train_sampler=None,
            **kwargs):

        """
        checking
        """
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        assert torch.npu.is_available(), "only support GPU version"
        device_num = len(self.args.gpus.split(","))

        """
        setup iterator
        """
        precise_bn_steps = 0 if not precise_bn else precise_bn_steps

        epoch_freeze_step = int(round(0.2*precise_bn_steps))
        epoch_train_steps = int(train_iter.batch_sampler.__len__() / epoch_div_factor)
        epoch_eval_steps = int(eval_iter.batch_sampler.__len__() / epoch_div_factor)
        if (train_iter.batch_sampler.__len__() - epoch_train_steps) > precise_bn_steps:
            # train iter is sufficient
            epoch_term_steps = epoch_train_steps + precise_bn_steps
        else:
            epoch_term_steps = epoch_train_steps
            epoch_train_steps = epoch_train_steps - precise_bn_steps
            logging.warning(">> using the last {} iter for computing the precise bathnorm.")

        """
        start the main loop
        """
        for i_epoch in range(epoch_start, epoch_end):
            if train_sampler is not None:
                train_sampler.set_epoch(i_epoch)

            self.callback_kwargs['epoch'] = i_epoch
            epoch_start_time = time.time()

            ###########
            # 1] TRAINING
            ###########
            metrics.reset()
            self.net.train()
            sum_batch_inst = 0
            sum_batch_elapse = 0.
            sum_update_elapse = 0
            batch_start_time = time.time()
            if self.args.master_node:
                logging.info("Start epoch {:d}, iter stride {:d}, train steps {:d}, eval steps: {:d} (with {} GPUs/NPUs node: {})".format( \
                            i_epoch, epoch_div_factor, epoch_train_steps, epoch_eval_steps, device_num, self.args.gpus))

            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')

            for i_batch, (data, target) in enumerate(train_iter):

                if i_batch == 5:
                    end = time.time()
                    if self.args.master_node:
                        logging.info(f"=============Start compute FPS=============")

                if i_batch >= epoch_term_steps:
                    break

                if precise_bn and i_batch == epoch_train_steps:
                    if self.args.master_node:
                        logging.info("Compute precise batchnorm: {} to {}.".format(epoch_train_steps, epoch_term_steps))
                    # TODO: better way to rsync running_mean / runing_var
                    self.save_checkpoint(epoch=i_epoch+1, optimizer_state=optimizer.state_dict())
                    while not os.path.exists(self.get_checkpoint_path(epoch=i_epoch+1)):
                        print("sleep 1 sec...")
                        time.sleep(1)
                    time.sleep(5)
                    self.load_checkpoint(epoch=i_epoch+1)
                    metrics.reset()

                self.callback_kwargs['batch'] = i_batch
                update_start_time = time.time()

                # saving prof
                if self.args.prof and i_epoch == 0:
                    with torch.autograd.profiler.profile(use_npu=True) as prof:
                        outputs, losses = self.forward(data, target)
                        # [backward]
                        if self.args.apex:
                            optimizer.zero_grad()
                            for loss in losses: 
                                with amp.scale_loss(loss, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            self.adjust_learning_rate(optimizer=optimizer, lr=lr_scheduler.update())
                            optimizer.step()
                        else:
                            if i_batch < epoch_train_steps:
                                optimizer.zero_grad()
                                for loss in losses: loss.backward()
                                self.adjust_learning_rate(optimizer=optimizer, lr=lr_scheduler.update())
                                optimizer.step()
                            elif i_batch < (epoch_term_steps - epoch_freeze_step):
                                optimizer.zero_grad()
                                for loss in losses: loss.backward()
                                self.adjust_learning_rate(optimizer=optimizer, lr=lr_scheduler.get_lr())
                                optimizer.step(visiable=["precise.bn"])
                            else:
                                pass    
                    logging.info(">>>prof: "+prof.key_averages().table(sort_by="self_cpu_time_total"))
                    prof.export_chrome_trace("output.prof")
                else:
                    # [forward] making next step
                    outputs, losses = self.forward(data, target)

                    # [backward]
                    if self.args.apex:
                        optimizer.zero_grad()
                        for loss in losses: 
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        self.adjust_learning_rate(optimizer=optimizer, lr=lr_scheduler.update())
                        optimizer.step()
                    else:
                        if i_batch < epoch_train_steps:
                            optimizer.zero_grad()
                            for loss in losses: loss.backward()
                            self.adjust_learning_rate(optimizer=optimizer, lr=lr_scheduler.update())
                            optimizer.step()
                        elif i_batch < (epoch_term_steps - epoch_freeze_step):
                            optimizer.zero_grad()
                            for loss in losses: loss.backward()
                            self.adjust_learning_rate(optimizer=optimizer, lr=lr_scheduler.get_lr())
                            optimizer.step(visiable=["precise.bn"])
                        else:
                            pass

                if i_batch >= 5:
                    # measure elapsed time TODO
                    batch_time.update(time.time() - end)
                    end = time.time()

                self.callback_kwargs['lr'] = lr_scheduler.get_lr()

                # [evaluation] update train metric
                metrics.update([output.data.cpu() for output in outputs],
                               target.cpu(),
                               [loss.data.cpu() for loss in losses])

                # timing each batch
                sum_batch_elapse += time.time() - batch_start_time
                sum_update_elapse += time.time() - update_start_time
                sum_batch_inst += 1

                if (i_batch % self.step_callback_freq) == 0:
                    name_value_prefix = 'tr-' if i_batch < epoch_train_steps else 'bn-'
                    self.callback_kwargs['namevals'] = metrics.get_name_value(prefix=name_value_prefix)
                    metrics.reset()
                    # speed monitor
                    self.callback_kwargs['batch_elapse'] = sum_batch_elapse / sum_batch_inst
                    self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_batch_inst
                    sum_update_elapse = 0
                    sum_batch_elapse = 0
                    sum_batch_inst = 0
                    # callbacks
                    self.step_end_callback()

                # save checkpoint in case of unexpected interrupt
                if (i_batch % 500) == 0 and i_batch < epoch_train_steps:
                    self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
                    self.callback_kwargs['optimizer_dict'] = optimizer.state_dict()
                    self.epoch_end_callback()

                # end of current train iter
                batch_start_time = time.time()

            if self.args.master_node:
                FPS = f"batch_size: {self.batch_size}, Time: {round(batch_time.avg, 3)}, FPS@all: {round( device_num * self.batch_size/batch_time.avg, 3)}"
                logging.info(FPS)


            ###########
            # 2] END OF EPOCH
            ###########
            self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
            self.callback_kwargs['optimizer_dict'] = optimizer.state_dict()
            self.epoch_end_callback()


            ###########
            # 3] Evaluation
            ###########
            if (eval_iter is not None) \
                and ((i_epoch+1) % max(1, int(self.save_checkpoint_freq/2))) == 0:
                if self.args.master_node:
                    logging.info("========================Start evaluating epoch {:d}:========================".format(i_epoch))

                metrics.reset()
                self.net.eval()
                sum_batch_elapse = 0.
                sum_batch_inst = 0
                sum_forward_elapse = 0.
                with torch.no_grad():
                    # if True:
                    batch_start_time = time.time()
                    for i_batch, (data, target) in enumerate(eval_iter):

                        forward_start_time = time.time()

                        outputs, losses = self.forward(data, target)

                        metrics.update([output.data.cpu() for output in outputs],
                                        target.cpu(),
                                       [loss.data.cpu() for loss in losses])

                        sum_forward_elapse += time.time() - forward_start_time
                        sum_batch_elapse += time.time() - batch_start_time
                        batch_start_time = time.time()
                        sum_batch_inst += 1

                        if i_batch >= epoch_eval_steps:
                            break

                    # evaluation callbacks
                    self.callback_kwargs['batch'] = sum_batch_inst
                    self.callback_kwargs['batch_elapse'] = sum_batch_elapse / sum_batch_inst
                    self.callback_kwargs['update_elapse'] = sum_forward_elapse / sum_batch_inst
                    self.callback_kwargs['namevals'] = metrics.get_name_value(prefix='ts-')
                    self.step_end_callback()

        if self.args.master_node:
            logging.info("Optimization done!")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = 10

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.batchsize = n
        
        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.batchsize):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.batchsize)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":

    """
    Test Static Model
    """
    '''
    import torch
    import torchvision

    logging.getLogger().setLevel(logging.DEBUG)

    resnet18 = torchvision.models.resnet18()
    net = static_model(net=resnet18, model_prefix="./exps/models/debug")

    net.save_checkpoint(epoch=10)
    net.load_checkpoint(epoch=10)
    '''

    """
    Test Typical Model
    """
    import sys
    import torch
    import torchvision
    sys.path.append(os.path.join("../../"))

    import metric
    from lr_scheduler import MultiFactorScheduler
    import data.dataiter_factory as dataiter_factory

    logging.getLogger().setLevel(logging.DEBUG)

    resume = False
    pretained = False

    ## -----------------
    resnet18 = torchvision.models.resnet18()
    import network
    logging.info(network.__file__)
    from network import symbol_builder
    sym_c3d, net_cfg = symbol_builder.get_symbol(name="c3d", num_classes=101)

    # settings for the optimization
    optimizer = torch.optim.SGD(sym_c3d.parameters(), lr=0.1,
                                momentum=0.9,
                                weight_decay=0.005)

    # initializatioln the dynamic model
    net = model(net=sym_c3d, optimizer=optimizer, criterion=torch.nn.CrossEntropyLoss().npu())

    # load the pretained model
    if resume:
        net.load_checkpoint(epoch=load_epoch)
    elif pretained:
        pretrained_model_state_dic = GetPretrainedModel(name='resnet')
        net.load_state(state_dic=pretrained_model_state_dic, strict=False)
    else:
        logging.info("Train from scratch using random initialization")

    # prepare opmitization
    metrics = metric.MetricList(metric.Accuracy(topk=1, name="acc-top1"),
                                metric.Accuracy(topk=5, name="acc-top5"))
    lr_scheduler = MultiFactorScheduler(steps=[300, 1000], base_lr=0.1, factor=0.1)


    tr_iter, ts_iter = dataiter_factory.creat(name='ucf101',
                                              data_root='../../dataset/UCF101',
                                              batch_size=1,
                                              )

    net.fit(iter_train=tr_iter, metrics_train=metrics,
            epoch_start=0, epoch_end=100,
            iter_eval=None, metrics_eval=None,
            lr_scheduler=lr_scheduler,)

    # print (net.get_checkpoint_path(epoch=19))
    # print (net.get_checkpoint_path(epoch=19))
