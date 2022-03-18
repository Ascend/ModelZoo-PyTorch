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

from concern.config import Configurable, State
from concern.log import Logger
from structure.builder import Builder
from structure.representers import *
from structure.measurers import *
from structure.visualizers import *
from data.data_loader import *
from data import *
from training.model_saver import ModelSaver
from training.checkpoint import Checkpoint
from training.optimizer_scheduler import OptimizerScheduler


class Structure(Configurable):
    builder = State()
    representer = State()
    measurer = State()
    visualizer = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    @property
    def model_name(self):
        return self.builder.model_name


class TrainSettings(Configurable):
    data_loader = State()
    model_saver = State()
    checkpoint = State()
    scheduler = State()
    epochs = State(default=10)

    def __init__(self, **kwargs):
        kwargs['cmd'].update(is_train=True)
        self.load_all(**kwargs)
        if 'epochs' in kwargs['cmd']:
            self.epochs = kwargs['cmd']['epochs']


class ValidationSettings(Configurable):
    data_loaders = State()
    visualize = State()
    interval = State(default=100)
    exempt = State(default=-1)

    def __init__(self, **kwargs):
        kwargs['cmd'].update(is_train=False)
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        self.visualize = cmd['visualize']


class EvaluationSettings(Configurable):
    data_loaders = State()
    visualize = State(default=True)
    resume = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)


class EvaluationSettings2(Configurable):
    structure = State()
    data_loaders = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)


class ShowSettings(Configurable):
    data_loader = State()
    representer = State()
    visualizer = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)


class Experiment(Configurable):
    structure = State(autoload=False)
    train = State()
    validation = State(autoload=False)
    evaluation = State(autoload=False)
    logger = State(autoload=True)

    def __init__(self, **kwargs):
        self.load('structure', **kwargs)

        cmd = kwargs.get('cmd', {})
        if 'name' not in cmd:
            cmd['name'] = self.structure.model_name

        self.load_all(**kwargs)
        self.distributed = cmd.get('distributed', False)
        self.amp = cmd.get('amp', True)
        self.local_rank = cmd.get('local_rank', 0)
        self.device_list = cmd.get("device_list", '0')
        self.device_id = cmd.get("device_id", '0')
        if cmd.get('validate', False):
            self.load('validation', **kwargs)
        else:
            self.validation = None
