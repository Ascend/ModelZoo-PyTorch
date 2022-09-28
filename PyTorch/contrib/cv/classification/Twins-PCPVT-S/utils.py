# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: utf-8 -*-
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
# import torch_npu

import torch.distributed as dist
#from main import useNPU
try:
    from apex import amp

    has_apex = True
    print("successfully import amp")
except ImportError:
    amp = None
    has_apex = False
    print("can not import amp from apex")


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None,use_npu=True):
        # fmt指定输出格式
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.use_npu = use_npu

    def update(self, value, n=1):
        # 更新指标：将新值加入窗口并将数量和总值进行更新
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        if self.use_npu:
            t = torch.tensor([self.count, self.total], dtype=torch.float,device='npu')
        else:
            t = torch.tensor([self.count, self.total], dtype=torch.float, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        # 计算窗口大小的中位数
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        # 计算窗口大小的均值
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        # 计算全局的均值
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        # 取得窗口的最后一个值
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class Fps(SmoothedValue):
    def __init__(self, window_size=20, fmt=None, num_devices=1, batch_size=1,use_npu=False):
        SmoothedValue.__init__(self, window_size, fmt,use_npu)
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.reserved = []


    def update(self, value, n=1):
        # 更新指标：将新值加入窗口并将数量和总值进行更新
        if self.count<5:
            self.reserved.append(value)
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def avg(self):
        # 计算窗口大小的均值
        if self.count <= 25:
            return 0
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return self.batch_size * self.num_devices / d.mean().item()


class MetricLogger(object):
    def __init__(self, delimiter="\t",use_npu=False):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.use_npu = use_npu

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        # iterate time 完成一次循环的时间
        iter_time = SmoothedValue(fmt='{avg:.4f}',use_npu=self.use_npu)
        # data_time 数据获取时间
        data_time = SmoothedValue(fmt='{avg:.4f}',use_npu=self.use_npu)
        # get info to compute the FPS

        num_devs = 1
        if hasApex():
            num_devs = get_world_size()
            print(f"number of world size:{num_devs}")
        print(f"worker info:{torch.utils.data.get_worker_info()}")
        batch_size = 1
        for obj in iterable:
            batch_size = len(obj[0])
            break
        print(f"batch size: {batch_size}")
        fps = Fps(window_size=20, fmt='{avg:.4f}', num_devices=num_devs,batch_size=batch_size,use_npu=self.use_npu)
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            # 展示当前epoch的进度
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'fps:{fps}',
            'data: {data}'
        ]
        if self.use_npu:
            if torch.npu.is_available():
                log_msg.append('max mem: {memory:.0f}')
        else:
            if torch.cuda.is_available():
                log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            fps.update(time.time() - end)
            # 每 print_freq 打印一次
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 计算预计还需要多长时间
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if self.use_npu:
                    if torch.npu.is_available():
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time), fps=str(fps),
                            memory=torch.npu.max_memory_allocated() / MB))
                    else:
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time), fps=str(fps)))
                else:
                    if torch.cuda.is_available():
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time), fps=str(fps),
                            memory=torch.cuda.max_memory_allocated() / MB))
                    else:
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time), fps=str(fps)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print
    builtin_print(f'is master:{is_master}')

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args,use_npu=False):
    # 函数里的每一行代码都会在每个进程上单独执行
    log = []  # 记录环境信息
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        log.extend([args.rank, args.world_size, args.gpu])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        if use_npu:
            args.gpu = args.rank % torch.npu.device_count()
        else:
            args.gpu = args.rank % torch.cuda.device_count()
        log.extend([args.rank, args.gpu])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    print(f"args.gpu:{args.gpu}")
    # 设置可以进行分布式训练
    args.distributed = True
    # 打印环境信息
    print(f"environment info: {log}")
    # 指定GPU显卡
    if use_npu:
        torch.npu.set_device(args.gpu)
        args.dist_backend = 'hccl'
    else:
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    # 分布式初始化
    if use_npu:
        torch.distributed.init_process_group(backend=args.dist_backend, #init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    else:
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)

    # 同步所有进程
    torch.distributed.barrier()
    # 禁止非master进程输出
    setup_for_distributed(args.rank == 0)


def initialize_amp(model, optimizer, opt_level="O1"):
    if has_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, combine_grad=True)
        return model, optimizer
    else:
        raise ImportError("can not import amp from apex")


def hasApex():
    return has_apex
