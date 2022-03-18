"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
import os
import sys
import os.path as osp

from .tools import mkdir_if_missing

__all__ = ['Logger', 'RankLogger']


class Logger(object):
    """Writes console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_

    Args:
        fpath (str): directory to save logging file.

    Examples::
       >>> import sys
       >>> import os
       >>> import os.path as osp
       >>> from torchreid.utils import Logger
       >>> save_dir = 'log/resnet50-softmax-market1501'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(args.save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class RankLogger(object):
    """Records the rank1 matching accuracy obtained for each
    test dataset at specified evaluation steps and provides a function
    to show the summarized results, which are convenient for analysis.

    Args:
        sources (str or list): source dataset name(s).
        targets (str or list): target dataset name(s).

    Examples::
        >>> from torchreid.utils import RankLogger
        >>> s = 'market1501'
        >>> t = 'market1501'
        >>> ranklogger = RankLogger(s, t)
        >>> ranklogger.write(t, 10, 0.5)
        >>> ranklogger.write(t, 20, 0.7)
        >>> ranklogger.write(t, 30, 0.9)
        >>> ranklogger.show_summary()
        >>> # You will see:
        >>> # => Show performance summary
        >>> # market1501 (source)
        >>> # - epoch 10   rank1 50.0%
        >>> # - epoch 20   rank1 70.0%
        >>> # - epoch 30   rank1 90.0%
        >>> # If there are multiple test datasets
        >>> t = ['market1501', 'dukemtmcreid']
        >>> ranklogger = RankLogger(s, t)
        >>> ranklogger.write(t[0], 10, 0.5)
        >>> ranklogger.write(t[0], 20, 0.7)
        >>> ranklogger.write(t[0], 30, 0.9)
        >>> ranklogger.write(t[1], 10, 0.1)
        >>> ranklogger.write(t[1], 20, 0.2)
        >>> ranklogger.write(t[1], 30, 0.3)
        >>> ranklogger.show_summary()
        >>> # You can see:
        >>> # => Show performance summary
        >>> # market1501 (source)
        >>> # - epoch 10   rank1 50.0%
        >>> # - epoch 20   rank1 70.0%
        >>> # - epoch 30   rank1 90.0%
        >>> # dukemtmcreid (target)
        >>> # - epoch 10   rank1 10.0%
        >>> # - epoch 20   rank1 20.0%
        >>> # - epoch 30   rank1 30.0%
    """

    def __init__(self, sources, targets):
        self.sources = sources
        self.targets = targets

        if isinstance(self.sources, str):
            self.sources = [self.sources]

        if isinstance(self.targets, str):
            self.targets = [self.targets]

        self.logger = {
            name: {
                'epoch': [],
                'rank1': []
            }
            for name in self.targets
        }

    def write(self, name, epoch, rank1):
        """Writes result.

        Args:
           name (str): dataset name.
           epoch (int): current epoch.
           rank1 (float): rank1 result.
        """
        self.logger[name]['epoch'].append(epoch)
        self.logger[name]['rank1'].append(rank1)

    def show_summary(self):
        """Shows saved results."""
        print('=> Show performance summary')
        for name in self.targets:
            from_where = 'source' if name in self.sources else 'target'
            print('{} ({})'.format(name, from_where))
            for epoch, rank1 in zip(
                self.logger[name]['epoch'], self.logger[name]['rank1']
            ):
                print('- epoch {}\t rank1 {:.1%}'.format(epoch, rank1))
