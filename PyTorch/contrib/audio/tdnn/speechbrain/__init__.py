#     Copyright 2021 Huawei Technologies Co., Ltd
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

""" Comprehensive speech processing toolkit
"""
import os
from .core import Stage, Brain, create_experiment_directory, parse_arguments
from . import alignment  # noqa
from . import dataio  # noqa
from . import decoders  # noqa
from . import lobes  # noqa
from . import lm  # noqa
from . import nnet  # noqa
from . import processing  # noqa
from . import tokenizers  # noqa
from . import utils  # noqa

with open(os.path.join(os.path.dirname(__file__), "version.txt")) as f:
    version = f.read().strip()

__all__ = [
    "Stage",
    "Brain",
    "create_experiment_directory",
    "parse_arguments",
]

__version__ = version
