# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from functools import partial

import mmcv

from mmocr.utils import list_from_file, list_to_file


def convert(closeset_line, merge_bg_others=False, ignore_idx=0, others_idx=25):
    """Convert line-json str of closeset to line-json str of openset. Note that
    this function is designed for closeset-wildreceipt to openset-wildreceipt.
    It may not be suitable to your own dataset.

    Args:
        closeset_line (str): The string to be deserialized to
            the closeset dictionary object.
        merge_bg_others (bool): If True, give the same label to "background"
            class and "others" class.
        ignore_idx (int): Index for ``ignore`` class.
        others_idx (int): Index for ``others`` class.
    """
    # Two labels at the same index of the following two lists
    # make up a key-value pair. For example, in wildreceipt,
    # closeset_key_inds[0] maps to "Store_name_key"
    # and closeset_value_inds[0] maps to "Store_addr_value".
    closeset_key_inds = list(range(2, others_idx, 2))
    closeset_value_inds = list(range(1, others_idx, 2))

    openset_node_label_mapping = {'bg': 0, 'key': 1, 'value': 2, 'others': 3}
    if merge_bg_others:
        openset_node_label_mapping['others'] = openset_node_label_mapping['bg']

    closeset_obj = json.loads(closeset_line)
    openset_obj = {
        'file_name': closeset_obj['file_name'],
        'height': closeset_obj['height'],
        'width': closeset_obj['width'],
        'annotations': []
    }

    edge_idx = 1
    label_to_edge = {}
    for anno in closeset_obj['annotations']:
        label = anno['label']
        if label == ignore_idx:
            anno['label'] = openset_node_label_mapping['bg']
            anno['edge'] = edge_idx
            edge_idx += 1
        elif label == others_idx:
            anno['label'] = openset_node_label_mapping['others']
            anno['edge'] = edge_idx
            edge_idx += 1
        else:
            edge = label_to_edge.get(label, None)
            if edge is not None:
                anno['edge'] = edge
                if label in closeset_key_inds:
                    anno['label'] = openset_node_label_mapping['key']
                elif label in closeset_value_inds:
                    anno['label'] = openset_node_label_mapping['value']
            else:
                tmp_key = 'key'
                if label in closeset_key_inds:
                    label_with_same_edge = closeset_value_inds[
                        closeset_key_inds.index(label)]
                elif label in closeset_value_inds:
                    label_with_same_edge = closeset_key_inds[
                        closeset_value_inds.index(label)]
                    tmp_key = 'value'
                edge_counterpart = label_to_edge.get(label_with_same_edge,
                                                     None)
                if edge_counterpart is not None:
                    anno['edge'] = edge_counterpart
                else:
                    anno['edge'] = edge_idx
                    edge_idx += 1
                anno['label'] = openset_node_label_mapping[tmp_key]
                label_to_edge[label] = anno['edge']

    openset_obj['annotations'] = closeset_obj['annotations']

    return json.dumps(openset_obj, ensure_ascii=False)


def process(closeset_file, openset_file, merge_bg_others=False, n_proc=10):
    closeset_lines = list_from_file(closeset_file)

    convert_func = partial(convert, merge_bg_others=merge_bg_others)

    openset_lines = mmcv.track_parallel_progress(
        convert_func, closeset_lines, nproc=n_proc)

    list_to_file(openset_file, openset_lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', help='Annotation file for closeset.')
    parser.add_argument('out_file', help='Annotation file for openset.')
    parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge two classes: "background" and "others" in closeset '
        'to one class in openset.')
    parser.add_argument(
        '--n_proc', type=int, default=10, help='Number of process.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    process(args.in_file, args.out_file, args.merge, args.n_proc)

    print('finish')


if __name__ == '__main__':
    main()
