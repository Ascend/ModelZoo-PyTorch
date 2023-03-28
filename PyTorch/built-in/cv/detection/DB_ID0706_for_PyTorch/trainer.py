#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import os
import time
import torch
from tqdm import tqdm

from experiment import Experiment
from data.data_loader import DistributedSampler
from apex import amp
from torch_npu.utils.profiler import Profile


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.init_device()
        if self.experiment.train.data_loader.seed is not None:
            seed_everything(self.experiment.train.data_loader.seed)
        self.structure = experiment.structure
        self.logger = experiment.logger
        self.model_saver = experiment.train.model_saver

        # FIXME: Hack the save model path into logger path
        self.model_saver.dir_path = self.logger.save_dir(
            self.model_saver.dir_path)
        self.current_lr = 0
        self.total = 0

    def init_device(self):
        if torch.npu.is_available():
            self.device = torch.device(f'npu:{self.experiment.device_id}')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(
            self.device, self.experiment.distributed, self.experiment.local_rank)
        return model

    def update_learning_rate(self, optimizer, epoch, step):
        lr = self.experiment.train.scheduler.learning_rate.get_learning_rate(
            epoch, step)

        for group in optimizer.param_groups:
            group['lr'] = lr
        self.current_lr = lr

    def train(self, profiling, collect_start, collect_end):
        self.logger.report_time('Start')
        self.logger.args(self.experiment)
        model = self.init_model()
        train_data_loader = self.experiment.train.data_loader.train_loader
        if self.experiment.validation:
            validation_loaders = self.experiment.validation.data_loaders
        self.steps = 0
        epoch = 0
        if self.experiment.train.checkpoint:
            self.experiment.train.checkpoint.restore_model(
                model, self.device, self.logger)
            epoch, iter_delta = self.experiment.train.checkpoint.restore_counter()
            self.steps = epoch * self.total + iter_delta

        # Init start epoch and iter
        optimizer = self.experiment.train.scheduler.create_optimizer(
            model.parameters())
        if self.experiment.amp:
            amp.register_float_function(torch, "sigmoid")
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=1024, combine_grad=True)
        self.logger.report_time('Init')
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        while True:
            self.logger.info('Training epoch ' + str(epoch))
            self.logger.epoch(epoch)
            self.total = len(train_data_loader)
            profiler = Profile(start_step=int(os.getenv("PROFILE_START_STEP", 10)),
                               profile_type=os.getenv("PROFILE_TYPE"))
            for batch in train_data_loader:
                data_time.update((time.time() - end) * 1000)
                self.update_learning_rate(optimizer, epoch, self.steps)
                self.logger.report_time("Data loading")
                if self.experiment.validation and\
                        self.steps % self.experiment.validation.interval == 0 and\
                        self.steps > self.experiment.validation.exempt:
                    self.validate(validation_loaders, model, epoch, self.steps)
                self.logger.report_time('Validating ')
                if self.logger.verbose:
                    torch.npu.synchronize()
                profiler.start()
                loss_item = self.train_step(model, optimizer, batch,
                                epoch=epoch, step=self.steps)
                profiler.end()
                losses.update(loss_item)
                if self.logger.verbose:
                    torch.npu.synchronize()
                self.logger.report_time('Forwarding ')
                self.model_saver.maybe_save_model(
                    model, epoch, self.steps, self.logger)
                self.steps += 1
                if self.steps == 9:
                    batch_time.reset()
                    data_time.reset()
                if self.total > 0 and self.steps % self.total == 2:
                    batch_time.reset()
                batch_time.update((time.time() - end) * 1000)
                if self.steps % self.experiment.logger.log_interval == 0:
                    fps = self.experiment.train.data_loader.batch_size * 1000 / batch_time.val
                    step_count = (self.steps - 1) % self.total + 1
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f}ms ({batch_time.avg:.3f}ms)\t' \
                        'Data {data_time.val:.3f}ms ({data_time.avg:.3f}ms)\t' \
                        'Fps {fps:.3f}\t' \
                        'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                        epoch, step_count, self.total, batch_time=batch_time,
                        data_time=data_time, fps=fps, loss=losses)
                    self.logger.info(msg)
                end = time.time()
            self.logger.info('npu id: {device_no:1d}' \
                  ' * FPS@all {fps:.3f}'.format(device_no=self.device.index, fps=self.experiment.train.data_loader.batch_size * 1000 / batch_time.avg))

            epoch += 1
            if epoch > self.experiment.train.epochs:
                if self.experiment.local_rank in [-1, 0]:
                    self.model_saver.save_checkpoint(model, 'final')
                    if self.experiment.validation:
                        self.validate(validation_loaders, model, epoch, self.steps)
                    self.logger.info('Training done')
                break
            iter_delta = 0

    def train_step(self, model, optimizer, batch, epoch, step, **kwards):
        # Temporary solution for huge memory usage by topk.
        # This should be deleted in future version.
        if step == 0 and epoch == 0:
            tmp = torch.rand(16, 16, 640, 640, dtype=torch.float32).npu()
            top, _ = torch.topk(tmp.view(-1).half(), 4000000)
            top, _ = torch.topk(tmp.view(-1).half(), 6000000)
        optimizer.zero_grad()
        results = model.forward(batch, training=True)
        if len(results) == 2:
            l, pred = results
            metrics = {}
        elif len(results) == 3:
            l, pred, metrics = results

        if isinstance(l, dict):
            line = []
            loss = torch.tensor(0.).npu()
            for key, l_val in l.items():
                loss += l_val.mean()
                line.append('loss_{0}:{1:.4f}'.format(key, l_val.mean()))
        else:
            loss = l
        if self.experiment.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        return loss.item()

    def validate(self, validation_loaders, model, epoch, step):
        all_matircs = {}
        model.eval()
        for name, loader in validation_loaders.items():
            if self.experiment.validation.visualize:
                metrics, vis_images = self.validate_step(
                    loader, model, True)
                self.logger.images(
                    os.path.join('vis', name), vis_images, step)
            else:
                metrics, vis_images = self.validate_step(loader, model, False)
            for _key, metric in metrics.items():
                key = name + '/' + _key
                if key in all_matircs:
                    all_matircs[key].update(metric.val, metric.count)
                else:
                    all_matircs[key] = metric

        for key, metric in all_matircs.items():
            self.logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))
        self.logger.metrics(epoch, self.steps, all_matircs)
        model.train()
        return all_matircs

    def validate_step(self, data_loader, model, visualize=False):
        raw_metrics = []
        vis_images = dict()
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            pred = model.forward(batch, training=False)
            output = self.structure.representer.represent(batch, pred)
            raw_metric, interested = self.structure.measurer.validate_measure(
                batch, output)
            raw_metrics.append(raw_metric)

            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.visualize(
                    batch, output, interested)
                vis_images.update(vis_image)
        metrics = self.structure.measurer.gather_measure(
            raw_metrics, self.logger)
        return metrics, vis_images

    def to_np(self, x):
        return x.cpu().data.numpy()
