# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Open-MMLab. All rights reserved.
from enum import Enum


class Priority(Enum):
    """Hook priority levels.

    +------------+------------+
    | Level      | Value      |
    +============+============+
    | HIGHEST    | 0          |
    +------------+------------+
    | VERY_HIGH  | 10         |
    +------------+------------+
    | HIGH       | 30         |
    +------------+------------+
    | NORMAL     | 50         |
    +------------+------------+
    | LOW        | 70         |
    +------------+------------+
    | VERY_LOW   | 90         |
    +------------+------------+
    | LOWEST     | 100        |
    +------------+------------+
    """

    HIGHEST = 0
    VERY_HIGH = 10
    HIGH = 30
    NORMAL = 50
    LOW = 70
    VERY_LOW = 90
    LOWEST = 100


def get_priority(priority):
    """Get priority value.

    Args:
        priority (int or str or :obj:`Priority`): Priority.

    Returns:
        int: The priority value.
    """
    if isinstance(priority, int):
        if priority < 0 or priority > 100:
            raise ValueError('priority must be between 0 and 100')
        return priority
    elif isinstance(priority, Priority):
        return priority.value
    elif isinstance(priority, str):
        return Priority[priority.upper()].value
    else:
        raise TypeError('priority must be an integer or Priority enum value')
