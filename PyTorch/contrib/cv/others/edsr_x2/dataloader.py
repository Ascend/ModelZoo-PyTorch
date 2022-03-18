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
import threading
import random

import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from torch.utils.data import BatchSampler
from torch.utils.data import _utils
from torch.utils.data.dataloader import _DataLoaderIter

from torch.utils.data._utils import collate
from torch.utils.data._utils import signal_handling
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data._utils import ExceptionWrapper
from torch.utils.data._utils import IS_WINDOWS
from torch.utils.data._utils.worker import ManagerWatchdog

from torch._six import queue


def _ms_loop(
    dataset,
    index_queue,
    data_queue,
    done_event,
    collate_fn,
    scale,
    seed,
    init_fn,
    worker_id,
):
    try:
        collate._use_shared_memory = True
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        data_queue.cancel_join_thread()

        if init_fn is not None:
            init_fn(worker_id)

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            if r is None:
                assert done_event.is_set()
                return
            elif done_event.is_set():
                continue

            idx, batch_indices = r
            try:
                idx_scale = 0
                if len(scale) > 1 and dataset.train:
                    idx_scale = random.randrange(0, len(scale))
                    dataset.set_scale(idx_scale)

                samples = collate_fn([dataset[i] for i in batch_indices])
                samples.append(idx_scale)
            except Exception:
                data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
            else:
                data_queue.put((idx, samples))
                del samples

    except KeyboardInterrupt:
        pass


class _MSDataLoaderIter(_DataLoaderIter):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout

        self.sample_iter = iter(self.batch_sampler)

        base_seed = torch.LongTensor(1).random_().item()

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.Queue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}
            self.done_event = multiprocessing.Event()

            base_seed = torch.LongTensor(1).random_()[0]

            self.index_queues = []
            self.workers = []
            for i in range(self.num_workers):
                index_queue = multiprocessing.Queue()
                index_queue.cancel_join_thread()
                w = multiprocessing.Process(
                    target=_ms_loop,
                    args=(
                        self.dataset,
                        index_queue,
                        self.worker_result_queue,
                        self.done_event,
                        self.collate_fn,
                        self.scale,
                        base_seed + i,
                        self.worker_init_fn,
                        i,
                    ),
                )
                w.daemon = True
                w.start()
                self.index_queues.append(index_queue)
                self.workers.append(w)

            if self.pin_memory:
                self.data_queue = queue.Queue()
                pin_memory_thread = threading.Thread(
                    target=_utils.pin_memory._pin_memory_loop,
                    args=(
                        self.worker_result_queue,
                        self.data_queue,
                        torch.cuda.current_device(),
                        self.done_event,
                    ),
                )
                pin_memory_thread.daemon = True
                pin_memory_thread.start()
                self.pin_memory_thread = pin_memory_thread
            else:
                self.data_queue = self.worker_result_queue

            _utils.signal_handling._set_worker_pids(
                id(self), tuple(w.pid for w in self.workers)
            )
            _utils.signal_handling._set_SIGCHLD_handler()
            self.worker_pids_set = True

            for _ in range(2 * self.num_workers):
                self._put_indices()


class MSDataLoader(DataLoader):
    def __init__(self, cfg, *args, **kwargs):
        super(MSDataLoader, self).__init__(*args, **kwargs, num_workers=cfg.workers)
        self.scale = cfg.scale

    def __iter__(self):
        return _MSDataLoaderIter(self)
