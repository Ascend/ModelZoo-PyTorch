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
#!/usr/bin/env python

from __future__ import print_function

import random
import time

from progress.bar import (Bar, ChargingBar, FillingSquaresBar,
                          FillingCirclesBar, IncrementalBar, PixelBar,
                          ShadyBar)
from progress.spinner import (Spinner, PieSpinner, MoonSpinner, LineSpinner,
                              PixelSpinner)
from progress.counter import Counter, Countdown, Stack, Pie


def sleep():
    t = 0.01
    t += t * random.uniform(-0.1, 0.1)  # Add some variance
    time.sleep(t)


for bar_cls in (Bar, ChargingBar, FillingSquaresBar, FillingCirclesBar):
    suffix = '%(index)d/%(max)d [%(elapsed)d / %(eta)d / %(eta_td)s]'
    bar = bar_cls(bar_cls.__name__, suffix=suffix)
    for i in bar.iter(range(200)):
        sleep()

for bar_cls in (IncrementalBar, PixelBar, ShadyBar):
    suffix = '%(percent)d%% [%(elapsed_td)s / %(eta)d / %(eta_td)s]'
    bar = bar_cls(bar_cls.__name__, suffix=suffix)
    for i in bar.iter(range(200)):
        sleep()

for spin in (Spinner, PieSpinner, MoonSpinner, LineSpinner, PixelSpinner):
    for i in spin(spin.__name__ + ' ').iter(range(100)):
        sleep()
    print()

for singleton in (Counter, Countdown, Stack, Pie):
    for i in singleton(singleton.__name__ + ' ').iter(range(100)):
        sleep()
    print()

bar = IncrementalBar('Random', suffix='%(index)d')
for i in range(100):
    bar.goto(random.randint(0, 100))
    sleep()
bar.finish()
