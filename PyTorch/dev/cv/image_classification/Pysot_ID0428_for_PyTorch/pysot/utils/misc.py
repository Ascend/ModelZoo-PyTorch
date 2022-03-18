#
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
#

# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from colorama import Fore, Style


__all__ = ['commit', 'describe']


def _exec(cmd):
    f = os.popen(cmd, 'r', 1)
    return f.read().strip()


def _bold(s):
    return "\033[1m%s\033[0m" % s


def _color(s):
    return f'{Fore.RED}{s}{Style.RESET_ALL}'


def _describe(model, lines=None, spaces=0):
    head = " " * spaces
    for name, p in model.named_parameters():
        if '.' in name:
            continue
        if p.requires_grad:
            name = _color(name)
        line = "{head}- {name}".format(head=head, name=name)
        lines.append(line)

    for name, m in model.named_children():
        space_num = len(name) + spaces + 1
        if m.training:
            name = _color(name)
        line = "{head}.{name} ({type})".format(
                head=head,
                name=name,
                type=m.__class__.__name__)
        lines.append(line)
        _describe(m, lines, space_num)


def commit():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    cmd = "cd {}; git log | head -n1 | awk '{{print $2}}'".format(root)
    commit = _exec(cmd)
    cmd = "cd {}; git log --oneline | head -n1".format(root)
    commit_log = _exec(cmd)
    return "commit : {}\n  log  : {}".format(commit, commit_log)


def describe(net, name=None):
    num = 0
    lines = []
    if name is not None:
        lines.append(name)
        num = len(name)
    _describe(net, lines, num)
    return "\n".join(lines)
