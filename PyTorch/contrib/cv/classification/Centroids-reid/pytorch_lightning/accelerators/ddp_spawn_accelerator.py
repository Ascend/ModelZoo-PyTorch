# Copyright The PyTorch Lightning team.
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
# limitations under the License
import os
import re
from typing import Any, List, Optional, Union

import torch
import torch.distributed as torch_distrib
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning import _logger as log
from pytorch_lightning.accelerators.accelerator import Accelerator, ReduceOp
from pytorch_lightning.cluster_environments import ClusterEnvironment
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.distributed import LightningDistributed
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.plugins.rpc_plugin import RPCPlugin
from pytorch_lightning.utilities import AMPType, HYDRA_AVAILABLE
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.distributed import (
    all_gather_ddp_if_available,
    find_free_network_port,
    rank_zero_only,
    rank_zero_warn,
    sync_ddp_if_available,
)
from pytorch_lightning.utilities.seed import seed_everything

if HYDRA_AVAILABLE:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd, to_absolute_path


class DDPSpawnAccelerator(Accelerator):

    def __init__(self,
                 trainer,
                 nprocs: int,
                 cluster_environment: Optional[ClusterEnvironment] = None,
                 ddp_plugin: Optional[DDPPlugin] = None):
        """
        Runs training using DDP using mp.spawn via manual launch (not cluster launch)

        Example::

            # default
            trainer = Trainer(accelerator=DDPSpawnAccelerator())

        """
        super().__init__(trainer, cluster_environment, ddp_plugin)
        self.mp_queue = None
        self.nprocs = nprocs
        self.dist = LightningDistributed()
        self.nickname = 'ddp'

    def setup(self, model):
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(find_free_network_port()))

        # pass in a state q
        smp = mp.get_context('spawn')
        self.mp_queue = smp.SimpleQueue()

        self.trainer.model = model

    def train(self):
        model = self.trainer.model

        # train in children process
        mp.spawn(self.ddp_train, nprocs=self.nprocs, args=(self.mp_queue, model,))

        # restore main state with best weights
        best_path = self.mp_queue.get()
        results = self.mp_queue.get()
        last_path = self.mp_queue.get()

        # recover the weights of the processes trained in the children
        self.__recover_child_process_weights(model, best_path, last_path)
        return results

    def ddp_train(self, process_idx, mp_queue, model, is_master=False, proc_offset=0):
        """
        Entry point for ddp

        Args:
            process_idx:
            mp_queue: multiprocessing queue
            model:
        """
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        # offset the process id if requested
        process_idx = process_idx + proc_offset

        # show progressbar only on progress_rank 0
        if (self.trainer.node_rank != 0 or process_idx != 0) and self.trainer.progress_bar_callback is not None:
            self.trainer.progress_bar_callback.disable()

        # determine which process we are and world size
        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.trainer.global_rank

        # Initialize cuda device
        self.init_device(process_idx, is_master)

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        model.trainer = self.trainer
        self.init_ddp_connection(
            self.trainer.global_rank,
            self.trainer.world_size,
            self.trainer.is_slurm_managing_tasks
        )

        if isinstance(self.ddp_plugin, RPCPlugin):
            if not self.ddp_plugin.is_main_rpc_process:
                self.ddp_plugin.on_accelerator_exit_rpc_process(self.trainer)
                self.ddp_plugin.exit_rpc_process()
                if self.ddp_plugin.return_after_exit_rpc_process:
                    return
            else:
                self.ddp_plugin.on_main_rpc_connection(self.trainer)

        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

        # on world_size=0 let everyone know training is starting
        if self.trainer.is_global_zero and not torch.distributed.is_initialized():
            log.info('-' * 100)
            log.info(f'distributed_backend={self.trainer.distributed_backend}')
            log.info(f'All DDP processes registered. Starting ddp with {self.trainer.world_size} processes')
            log.info('-' * 100)

        # call sync_bn before .cuda(), configure_apex and configure_ddp
        if self.trainer.sync_batchnorm:
            model = self.configure_sync_batchnorm(model)

        # move the model to the correct device
        self.model_to_device(model)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.setup_optimizers(model)

        self.ddp_plugin.on_after_setup_optimizers(self.trainer)

        # set model properties before going into wrapper
        self.trainer.model_connector.copy_trainer_model_properties(model)

        # 16-bit
        model = self.trainer.precision_connector.connect(model)

        # device ids change depending on the DDP setup
        device_ids = self.get_device_ids()

        # allow user to configure ddp
        model = self.configure_ddp(model, device_ids)

        # set up training routine
        self.trainer.train_loop.setup_training(model)

        # train or test
        results = self.train_or_test()

        # get original model
        model = self.trainer.get_model()

        # persist info in ddp_spawn
        self.transfer_distrib_spawn_state_on_fit_end(model, mp_queue, results)

        # clean up memory
        torch.cuda.empty_cache()

    def set_world_ranks(self, process_idx):
        self.trainer.local_rank = process_idx
        self.trainer.global_rank = self.trainer.node_rank * self.trainer.num_processes + process_idx
        self.trainer.world_size = self.trainer.num_nodes * self.trainer.num_processes

    def init_device(self, process_idx, is_master):
        gpu_idx = self.trainer.data_parallel_device_ids[self.trainer.local_rank]
        self.trainer.root_gpu = gpu_idx
        torch.cuda.set_device(self.trainer.root_gpu)

    def model_to_device(self, model):
        model.cuda(self.trainer.root_gpu)

    def get_device_ids(self):
        device_ids = [self.trainer.root_gpu]
        return device_ids

    def training_step(self, args):
        return self._step(args)

    def validation_step(self, args):
        return self._step(args)

    def test_step(self, args):
        return self._step(args)

    def _step(self, args):
        args = self.ddp_plugin.on_before_forward(self.trainer.get_model(), *args)
        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = self.trainer.model(*args)
        else:
            output = self.trainer.model(*args)
        return output

    def barrier(self, name: Optional[str] = None):
        if torch_distrib.is_initialized():
            torch_distrib.barrier()

    def early_stopping_should_stop(self, pl_module):
        stop = torch.tensor(int(self.trainer.should_stop), device=pl_module.device)
        torch_distrib.all_reduce(stop, op=torch_distrib.reduce_op.SUM)
        torch_distrib.barrier()
        should_stop = stop == self.trainer.world_size
        return should_stop

    def broadcast(self, obj, src=0):
        return self.dist.broadcast(obj)

    def __recover_child_process_weights(self, model, best_path, last_path):
        # transfer back the best path to the trainer
        if self.trainer.checkpoint_callback:
            self.trainer.checkpoint_callback.best_model_path = best_path
        # todo, pass also best score

        # load last weights
        if last_path is not None and not self.trainer.testing:
            ckpt = pl_load(last_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt)

        self.trainer.model = model

    def transfer_distrib_spawn_state_on_fit_end(self, model, mp_queue, results):
        best_model_path = None
        if self.trainer.checkpoint_callback is not None:
            best_model_path = self.trainer.checkpoint_callback.best_model_path

        if self.trainer.global_rank == 0 and mp_queue is not None:
            rank_zero_warn('cleaning up ddp environment...')
            # todo, pass complete checkpoint as state dictionary
            mp_queue.put(best_model_path)
            mp_queue.put(results)

            # save the last weights
            last_path = None
            if not self.trainer.testing and best_model_path is not None and len(best_model_path) > 0:
                last_path = re.sub('.ckpt', '.tmp_end.ckpt', best_model_path)
                atomic_save(model.state_dict(), last_path)
            mp_queue.put(last_path)

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ) -> DistributedDataParallel:
        self.ddp_plugin.device_ids = device_ids
        model = self.ddp_plugin.configure_ddp(model, device_ids)
        return model

    def configure_sync_batchnorm(self, model: LightningModule) -> LightningModule:
        """
        Add global batchnorm for a model spread across multiple GPUs and nodes.

        Override to synchronize batchnorm between specific process groups instead
        of the whole world or use a different sync_bn like `apex`'s version.

        Args:
            model: pointer to current :class:`LightningModule`.

        Return:
            LightningModule with batchnorm layers synchronized between process groups
        """
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None)

        return model

    def sync_tensor(self,
                    tensor: Union[torch.Tensor],
                    group: Optional[Any] = None,
                    reduce_op: Optional[Union[ReduceOp, str]] = None) -> torch.Tensor:
        return sync_ddp_if_available(tensor, group, reduce_op)

    def all_gather(self, tensor: Union[torch.Tensor], group: Optional[Any] = None, sync_grads: bool = False):
        """
        Function to gather a tensor from several distributed processes

        Args:
            tensor: tensor of shape (batch, ...)
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for all_gather op

        Return:
            A tensor of shape (world_size, batch, ...)
        """
        return all_gather_ddp_if_available(tensor, group=group, sync_grads=sync_grads)

    def get_reference_model(self, model) -> LightningModule:
        return self.ddp_plugin.get_model_from_plugin(model)

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(
            num_replicas=self.trainer.num_nodes * self.trainer.num_processes,
            rank=self.trainer.global_rank
        )
        if self.ddp_plugin is not None:
            distributed_sampler_kwargs = self.ddp_plugin.distributed_sampler_kwargs(distributed_sampler_kwargs)
        return distributed_sampler_kwargs

    @property
    def require_distributed_sampler(self):
        return True
