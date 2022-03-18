#!/usr/bin/env python3
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
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import torch
import os
import re


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['model']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k].div_(num_models)
    new_state['model'] = averaged_params
    return new_state


def last_n_checkpoints(paths, n, update_based, upper_bound=None):
    assert len(paths) == 1
    path = paths[0]
    if update_based:
        pt_regexp = re.compile(r'checkpoint_\d+_(\d+)\.pt')
    else:
        pt_regexp = re.compile(r'checkpoint(\d+)\.pt')
    files = os.listdir(path)

    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        if m is not None:
            sort_key = int(m.group(1))
            if upper_bound is None or sort_key <= upper_bound:
                entries.append((sort_key, m.group(0)))
    if len(entries) < n:
        raise Exception('Found {} checkpoint files but need at least {}', len(entries), n)
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)[:n]]


def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')
    num_group = parser.add_mutually_exclusive_group()
    num_group.add_argument('--num-epoch-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_xx.pt in the path specified by input, '
                           'and average last this many of them.')
    num_group.add_argument('--num-update-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_ee_xx.pt in the path specified by input, '
                           'and average last this many of them.')
    parser.add_argument('--checkpoint-upper-bound', type=int,
                        help='when using --num-epoch-checkpoints, this will set an upper bound on which checkpoint to use, '
                        'e.g., with --num-epoch-checkpoints=10 --checkpoint-upper-bound=50, checkpoints 41-50 would be averaged.')
    # fmt: on
    args = parser.parse_args()
    print(args)

    num = None
    is_update_based = False
    if args.num_update_checkpoints is not None:
        num = args.num_update_checkpoints
        is_update_based = True
    elif args.num_epoch_checkpoints is not None:
        num = args.num_epoch_checkpoints

    assert args.checkpoint_upper_bound is None or args.num_epoch_checkpoints is not None, \
        '--checkpoint-upper-bound requires --num-epoch-checkpoints'
    assert args.num_epoch_checkpoints is None or args.num_update_checkpoints is None, \
        'Cannot combine --num-epoch-checkpoints and --num-update-checkpoints'

    if num is not None:
        args.inputs = last_n_checkpoints(
            args.inputs, num, is_update_based, upper_bound=args.checkpoint_upper_bound,
        )
        print('averaging checkpoints: ', args.inputs)

    new_state = average_checkpoints(args.inputs)
    torch.save(new_state, args.output)
    print('Finished writing averaged checkpoint to {}.'.format(args.output))


if __name__ == '__main__':
    main()
