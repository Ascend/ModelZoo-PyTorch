
# Copyright 2020 Huawei Technologies Co., Ltd
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

"""
Lightning makes multi-gpu training and 16 bit training trivial.

.. note:: None of the flags below require changing anything about your lightningModel definition.

Choosing a backend
==================

Lightning supports two backends. DataParallel and DistributedDataParallel.
 Both can be used for single-node multi-GPU training.
 For multi-node training you must use DistributedDataParallel.

DataParallel (dp)
-----------------

Splits a batch across multiple GPUs on the same node. Cannot be used for multi-node training.

DistributedDataParallel (ddp)
-----------------------------

Trains a copy of the model on each GPU and only syncs gradients. If used with DistributedSampler, each GPU trains
on a subset of the full dataset.

DistributedDataParallel-2 (ddp2)
--------------------------------

Works like DDP, except each node trains a single copy of the model using ALL GPUs on that node.
 Very useful when dealing with negative samples, etc...

You can toggle between each mode by setting this flag.

.. code-block:: python

    # DEFAULT (when using single GPU or no GPUs)
    trainer = Trainer(distributed_backend=None)

    # Change to DataParallel (gpus > 1)
    trainer = Trainer(distributed_backend='dp')

    # change to distributed data parallel (gpus > 1)
    trainer = Trainer(distributed_backend='ddp')

    # change to distributed data parallel (gpus > 1)
    trainer = Trainer(distributed_backend='ddp2')

If you request multiple nodes, the back-end will auto-switch to ddp.
 We recommend you use DistributedDataparallel even for single-node multi-GPU training.
 It is MUCH faster than DP but *may* have configuration issues depending on your cluster.

For a deeper understanding of what lightning is doing, feel free to read this
 `guide <https://medium.com/@_willfalcon/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565>`_.

Distributed and 16-bit precision
--------------------------------

Due to an issue with apex and DistributedDataParallel (PyTorch and NVIDIA issue), Lightning does
 not allow 16-bit and DP training. We tried to get this to work, but it's an issue on their end.

Below are the possible configurations we support.

+-------+---------+----+-----+---------+------------------------------------------------------------+
| 1 GPU | 1+ GPUs | DP | DDP | 16-bit  | command                                                    |
+=======+=========+====+=====+=========+============================================================+
| Y     |         |    |     |         | `Trainer(gpus=1)`                                          |
+-------+---------+----+-----+---------+------------------------------------------------------------+
| Y     |         |    |     | Y       | `Trainer(gpus=1, use_amp=True)`                            |
+-------+---------+----+-----+---------+------------------------------------------------------------+
|       | Y       | Y  |     |         | `Trainer(gpus=k, distributed_backend='dp')`                |
+-------+---------+----+-----+---------+------------------------------------------------------------+
|       | Y       |    | Y   |         | `Trainer(gpus=k, distributed_backend='ddp')`               |
+-------+---------+----+-----+---------+------------------------------------------------------------+
|       | Y       |    | Y   | Y       | `Trainer(gpus=k, distributed_backend='ddp', use_amp=True)` |
+-------+---------+----+-----+---------+------------------------------------------------------------+

You also have the option of specifying which GPUs to use by passing a list:

.. code-block:: python

    # DEFAULT (int) specifies how many GPUs to use.
    Trainer(gpus=k)

    # Above is equivalent to
    Trainer(gpus=list(range(k)))

    # You specify which GPUs (don't use if running on cluster)
    Trainer(gpus=[0, 1])

    # can also be a string
    Trainer(gpus='0, 1')

    # can also be -1 or '-1', this uses all available GPUs
    # this is equivalent to list(range(torch.cuda.available_devices()))
    Trainer(gpus=-1)


CUDA flags
----------

CUDA flags make certain GPUs visible to your script.
 Lightning sets these for you automatically, there's NO NEED to do this yourself.

.. code-block:: python

    # lightning will set according to what you give the trainer
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


However, when using a cluster, Lightning will NOT set these flags (and you should not either).
 SLURM will set these for you.

16-bit mixed precision
----------------------

16 bit precision can cut your memory footprint by half. If using volta architecture GPUs
 it can give a dramatic training speed-up as well.
 First, install apex (if install fails, look `here <https://github.com/NVIDIA/apex>`_::

    $ git clone https://github.com/NVIDIA/apex
    $ cd apex

    # ------------------------
    # OPTIONAL: on your cluster you might need to load cuda 10 or 9
    # depending on how you installed PyTorch

    # see available modules
    module avail

    # load correct cuda before install
    module load cuda-10.0
    # ------------------------

    # make sure you've loaded a cuda version > 4.0 and < 7.0
    module load gcc-6.1.0

    $ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


then set this use_amp to True.::

    # DEFAULT
    trainer = Trainer(amp_level='O2', use_amp=False)


Single-gpu
----------

Make sure you're on a GPU machine.::

    # DEFAULT
    trainer = Trainer(gpus=1)

Multi-gpu
---------

Make sure you're on a GPU machine. You can set as many GPUs as you want.
 In this setting, the model will run on all 8 GPUs at once using DataParallel under the hood.

.. code-block:: python

    # to use DataParallel
    trainer = Trainer(gpus=8, distributed_backend='dp')

    # RECOMMENDED use DistributedDataParallel
    trainer = Trainer(gpus=8, distributed_backend='ddp')

Custom device selection
-----------------------

The number of GPUs can also be selected with a list of indices or a string containing
a comma separated list of GPU ids.
The table below lists examples of possible input formats and how they are interpreted by Lightning.
Note in particular the difference between `gpus=0`, `gpus=[0]` and `gpus="0"`.

+---------------+-----------+---------------------+---------------------------------+
| `gpus`        | Type      | Parsed              | Meaning                         |
+===============+===========+=====================+=================================+
| None          | NoneType  | None                | CPU                             |
+---------------+-----------+---------------------+---------------------------------+
| 0             | int       | None                | CPU                             |
+---------------+-----------+---------------------+---------------------------------+
| 3             | int       | [0, 1, 2]           | first 3 GPUs                    |
+---------------+-----------+---------------------+---------------------------------+
| -1            | int       | [0, 1, 2, ...]      | all available GPUs              |
+---------------+-----------+---------------------+---------------------------------+
| [0]           | list      | [0]                 | GPU 0                           |
+---------------+-----------+---------------------+---------------------------------+
| [1, 3]        | list      | [1, 3]              | GPUs 1 and 3                    |
+---------------+-----------+---------------------+---------------------------------+
| "0"           | str       | [0]                 | GPU 0                           |
+---------------+-----------+---------------------+---------------------------------+
| "3"           | str       | [3]                 | GPU 3                           |
+---------------+-----------+---------------------+---------------------------------+
| "1, 3"        | str       | [1, 3]              | GPUs 1 and 3                    |
+---------------+-----------+---------------------+---------------------------------+
| "-1"          | str       | [0, 1, 2, ...]      | all available GPUs              |
+---------------+-----------+---------------------+---------------------------------+


Multi-node
----------

Multi-node training is easily done by specifying these flags.

.. code-block:: python

    # train on 12*8 GPUs
    trainer = Trainer(gpus=8, num_nodes=12, distributed_backend='ddp')


You must configure your job submission script correctly for the trainer to work.
 Here is an example script for the above trainer configuration.

.. code-block:: bash

    #!/bin/bash -l

    # SLURM SUBMIT SCRIPT
    #SBATCH --nodes=12
    #SBATCH --gres=gpu:8
    #SBATCH --ntasks-per-node=8
    #SBATCH --mem=0
    #SBATCH --time=0-02:00:00

    # activate conda env
    conda activate my_env

    # -------------------------
    # OPTIONAL
    # -------------------------
    # debugging flags (optional)
    # export NCCL_DEBUG=INFO
    # export PYTHONFAULTHANDLER=1

    # PyTorch comes with prebuilt NCCL support... but if you have issues with it
    # you might need to load the latest version from your  modules
    # module load NCCL/2.4.7-1-cuda.10.0

    # on your cluster you might need these:
    # set the network interface
    # export NCCL_SOCKET_IFNAME=^docker0,lo
    # -------------------------

    # random port between 12k and 20k
    export MASTER_PORT=$((12000 + RANDOM % 20000))

    # run script from above
    python my_main_file.py

.. note:: When running in DDP mode, any errors in your code will show up as an NCCL issue.
 Set the `NCCL_DEBUG=INFO` flag to see the ACTUAL error.

Finally, make sure to add a distributed sampler to your dataset. The distributed sampler copies a
 portion of your dataset onto each GPU. (World_size = gpus_per_node * nb_nodes).

.. code-block:: python

    # ie: this:
    dataset = myDataset()
    dataloader = Dataloader(dataset)

    # becomes:
    dataset = myDataset()
    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = Dataloader(dataset, sampler=dist_sampler)


Auto-slurm-job-submission
-------------------------

Instead of manually building SLURM scripts, you can use the
`SlurmCluster object <https://williamfalcon.github.io/test-tube/hpc/SlurmCluster>`_
to do this for you. The SlurmCluster can also run a grid search if you pass
in a `HyperOptArgumentParser
<https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser>`_.

Here is an example where you run a grid search of 9 combinations of hyperparams.
The full examples are
`here <https://git.io/Jv87p>`_.

.. code-block:: python

    # grid search 3 values of learning rate and 3 values of number of layers for your net
    # this generates 9 experiments (lr=1e-3, layers=16), (lr=1e-3, layers=32),
    #  (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)
    parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)
    parser.opt_list('--learning_rate', default=0.001, type=float,
                    options=[1e-3, 1e-2, 1e-1], tunable=True)
    parser.opt_list('--layers', default=1, type=float, options=[16, 32, 64], tunable=True)
    hyperparams = parser.parse_args()

    # Slurm cluster submits 9 jobs, each with a set of hyperparams
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path='/some/path/to/save',
    )

    # OPTIONAL FLAGS WHICH MAY BE CLUSTER DEPENDENT
    # which interface your nodes use for communication
    cluster.add_command('export NCCL_SOCKET_IFNAME=^docker0,lo')

    # see output of the NCCL connection process
    # NCCL is how the nodes talk to each other
    cluster.add_command('export NCCL_DEBUG=INFO')

    # setting a master port here is a good idea.
    cluster.add_command('export MASTER_PORT=%r' % PORT)

    # ************** DON'T FORGET THIS ***************
    # MUST load the latest NCCL version
    cluster.load_modules(['NCCL/2.4.7-1-cuda.10.0'])

    # configure cluster
    cluster.per_experiment_nb_nodes = 12
    cluster.per_experiment_nb_gpus = 8

    cluster.add_slurm_cmd(cmd='ntasks-per-node', value=8, comment='1 task per gpu')

    # submit a script with 9 combinations of hyper params
    # (lr=1e-3, layers=16), (lr=1e-3, layers=32), (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)
    cluster.optimize_parallel_cluster_gpu(
        main,
        nb_trials=9, # how many permutations of the grid search to run
        job_name='name_for_squeue'
    )


The other option is that you generate scripts on your own via a bash command or use another library...

Self-balancing architecture
---------------------------

Here lightning distributes parts of your module across available GPUs to optimize for speed and memory.

"""

from abc import ABC, abstractmethod
import logging as log
import os
import signal

import torch

from pytorch_lightning.overrides.data_parallel import (
    LightningDistributedDataParallel,
    LightningDataParallel,
)
from pytorch_lightning.utilities.debugging import MisconfigurationException

try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class TrainerDPMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    on_gpu: bool
    use_dp: bool
    use_ddp2: bool
    use_ddp: bool
    use_amp: bool
    testing: bool
    single_gpu: bool
    root_gpu: ...
    amp_level: str
    precision: ...
    current_tpu_idx: ...
    proc_rank: int
    tpu_local_core_rank: int
    tpu_global_core_rank: int
    use_tpu: bool
    data_parallel_device_ids: ...

    @abstractmethod
    def run_pretrain_routine(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def init_optimizers(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def copy_trainer_model_properties(self, model):
        if isinstance(model, LightningDataParallel):
            ref_model = model.module
        elif isinstance(model, LightningDistributedDataParallel):
            ref_model = model.module
        else:
            ref_model = model

        for m in [model, ref_model]:
            m.trainer = self
            m.on_gpu = self.on_gpu
            m.use_dp = self.use_dp
            m.use_ddp2 = self.use_ddp2
            m.use_ddp = self.use_ddp
            m.use_amp = self.use_amp
            m.testing = self.testing
            m.single_gpu = self.single_gpu
            m.use_tpu = self.use_tpu
            m.tpu_local_core_rank = self.tpu_local_core_rank
            m.tpu_global_core_rank = self.tpu_global_core_rank

    def transfer_batch_to_tpu(self, batch):
        return self.__transfer_data_to_device(batch, device='tpu')

    def transfer_batch_to_gpu(self, batch, gpu_id):
        return self.__transfer_data_to_device(batch, device='npu', gpu_id=gpu_id)

    def __transfer_data_to_device(self, batch, device, gpu_id=None):
        if device == 'tpu' and XLA_AVAILABLE:
            # base case: object can be directly moved using `to`
            if callable(getattr(batch, 'to', None)):
                return batch.to(xm.xla_device())

        if device == 'npu':
            # base case: object can be directly moved using `cuda` or `to`
            if callable(getattr(batch, 'npu', None)):
                return batch.npu(gpu_id)

            if callable(getattr(batch, 'to', None)):
                return batch.to(torch.device('npu', gpu_id))

        # when list
        if isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.__transfer_data_to_device(x, device, gpu_id)
            return batch

        # when tuple
        if isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.__transfer_data_to_device(x, device, gpu_id)
            return tuple(batch)

        # when dict
        if isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.__transfer_data_to_device(v, device, gpu_id)

            return batch

        # nothing matches, return the value as is without transform
        return batch

    def single_gpu_train(self, model):
        model.npu(self.root_gpu)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        if self.use_amp:
            # An example
            model, optimizers = model.configure_apex(amp, model, self.optimizers, self.amp_level)
            self.optimizers = optimizers

        self.run_pretrain_routine(model)

    def tpu_train(self, tpu_core_idx, model):
        # put model on tpu
        model.to(xm.xla_device())

        # get the appropriate tpu ranks
        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()

        # avoid duplicating progress bar
        self.show_progress_bar = self.show_progress_bar and self.tpu_global_core_rank == 0

        # track current tpu
        self.current_tpu_idx = tpu_core_idx
        self.proc_rank = self.tpu_local_core_rank

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        # init 16 bit for TPU
        if self.precision == 16:
            os.environ['XLA_USE_BF16'] = str(1)

        m = f'INIT TPU local core: {self.tpu_local_core_rank}, ' \
            f'global rank: {self.tpu_global_core_rank}'
        log.info(m)

        # continue training routine
        self.run_pretrain_routine(model)

        self.save_spawn_weights(model)

    def dp_train(self, model):

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        model.cuda(self.root_gpu)

        # check for this bug (amp + dp + !01 doesn't work)
        # https://github.com/NVIDIA/apex/issues/227
        if self.use_dp and self.use_amp:
            if self.amp_level == 'O2':  # pragma: no cover
                m = f"""
                Amp level {self.amp_level} with DataParallel is not supported.
                See this note from NVIDIA for more info: https://github.com/NVIDIA/apex/issues/227.
                We recommend you switch to ddp if you want to use amp
                """
                raise MisconfigurationException(m)
            else:
                model, optimizers = model.configure_apex(amp, model, self.optimizers, self.amp_level)

        # create list of device ids
        device_ids = self.data_parallel_device_ids
        if isinstance(device_ids, int):
            device_ids = list(range(device_ids))

        model = LightningDataParallel(model, device_ids=device_ids)

        self.run_pretrain_routine(model)


def normalize_parse_gpu_string_input(s):
    if isinstance(s, str):
        if s == '-1':
            return -1
        else:
            return [int(x.strip()) for x in s.split(',')]
    else:
        return s


def get_all_available_gpus():
    """
    :return: a list of all available gpus
    """
    return list(range(torch.cuda.device_count()))


def check_gpus_data_type(gpus):
    """
    :param gpus: gpus parameter as passed to the Trainer
        Function checks that it is one of: None, Int, String or List
        Throws otherwise
    :return: return unmodified gpus variable
    """

    if gpus is not None and type(gpus) not in (int, str, list):
        raise MisconfigurationException("GPUs must be int, string or list of ints or None.")


def normalize_parse_gpu_input_to_list(gpus):
    assert gpus is not None
    if isinstance(gpus, list):
        return gpus

    # must be an int
    if not gpus:  # gpus==0
        return None
    if gpus == -1:
        return get_all_available_gpus()

    return list(range(gpus))


def sanitize_gpu_ids(gpus):
    """
    :param gpus: list of ints corresponding to GPU indices
        Checks that each of the GPUs in the list is actually available.
        Throws if any of the GPUs is not available.
    :return: unmodified gpus variable
    """
    all_available_gpus = get_all_available_gpus()
    for gpu in gpus:
        if gpu not in all_available_gpus:
            message = f"""
            You requested GPUs: {gpus}
            But your machine only has: {all_available_gpus}
            """
            raise MisconfigurationException(message)
    return gpus


def parse_gpu_ids(gpus):
    """
    :param gpus: Int, string or list
        An int -1 or string '-1' indicate that all available GPUs should be used.
        A list of ints or a string containing list of comma separated integers
        indicates specific GPUs to use
        An int 0 means that no GPUs should be used
        Any int N > 0 indicates that GPUs [0..N) should be used.
    :return: List of gpus to be used

        If no GPUs are available but the value of gpus variable indicates request for GPUs
        then a misconfiguration exception is raised.
    """

    # Check that gpus param is None, Int, String or List
    check_gpus_data_type(gpus)

    # Handle the case when no gpus are requested
    if gpus is None or isinstance(gpus, int) and gpus == 0:
        return None

    # We know user requested GPUs therefore if some of the
    # requested GPUs are not available an exception is thrown.

    gpus = normalize_parse_gpu_string_input(gpus)
    gpus = normalize_parse_gpu_input_to_list(gpus)
    # gpus = sanitize_gpu_ids(gpus)

    if not gpus:
        raise MisconfigurationException("GPUs requested but none are available.")
    return gpus


def determine_root_gpu_device(gpus):
    """
    :param gpus: non empty list of ints representing which gpus to use
    :return: designated root GPU device
    """
    if gpus is None:
        return None

    assert isinstance(gpus, list), "gpus should be a list"
    assert len(gpus) > 0, "gpus should be a non empty list"

    # set root gpu
    root_gpu = gpus[0]

    return root_gpu
