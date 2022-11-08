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
from mmocr.models.builder import CONVERTORS
from mmocr.utils import list_from_file


@CONVERTORS.register_module()
class BaseConvertor:
    """Convert between text, index and tensor for text recognize pipeline.

    Args:
        dict_type (str): Type of dict, options are 'DICT36', 'DICT37', 'DICT90'
            and 'DICT91'.
        dict_file (None|str): Character dict file path. If not none,
            the dict_file is of higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, the list
            is of higher priority than dict_type, but lower than dict_file.
    """
    start_idx = end_idx = padding_idx = 0
    unknown_idx = None
    lower = False

    dicts = dict(
        DICT36=tuple('0123456789abcdefghijklmnopqrstuvwxyz'),
        DICT90=tuple('0123456789abcdefghijklmnopqrstuvwxyz'
                     'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()'
                     '*+,-./:;<=>?@[\\]_`~'),
        # With space character
        DICT37=tuple('0123456789abcdefghijklmnopqrstuvwxyz '),
        DICT91=tuple('0123456789abcdefghijklmnopqrstuvwxyz'
                     'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()'
                     '*+,-./:;<=>?@[\\]_`~ '))

    def __init__(self, dict_type='DICT90', dict_file=None, dict_list=None):
        assert dict_file is None or isinstance(dict_file, str)
        assert dict_list is None or isinstance(dict_list, list)
        self.idx2char = []
        if dict_file is not None:
            for line_num, line in enumerate(list_from_file(dict_file)):
                line = line.strip('\r\n')
                if len(line) > 1:
                    raise ValueError('Expect each line has 0 or 1 character, '
                                     f'got {len(line)} characters '
                                     f'at line {line_num + 1}')
                if line != '':
                    self.idx2char.append(line)
        elif dict_list is not None:
            self.idx2char = list(dict_list)
        else:
            if dict_type in self.dicts:
                self.idx2char = list(self.dicts[dict_type])
            else:
                raise NotImplementedError(f'Dict type {dict_type} is not '
                                          'supported')

        assert len(set(self.idx2char)) == len(self.idx2char), \
            'Invalid dictionary: Has duplicated characters.'

        self.char2idx = {char: idx for idx, char in enumerate(self.idx2char)}

    def num_classes(self):
        """Number of output classes."""
        return len(self.idx2char)

    def str2idx(self, strings):
        """Convert strings to indexes.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        """
        assert isinstance(strings, list)

        indexes = []
        for string in strings:
            if self.lower:
                string = string.lower()
            index = []
            for char in string:
                char_idx = self.char2idx.get(char, self.unknown_idx)
                if char_idx is None:
                    raise Exception(f'Chararcter: {char} not in dict,'
                                    f' please check gt_label and use'
                                    f' custom dict file,'
                                    f' or set "with_unknown=True"')
                index.append(char_idx)
            indexes.append(index)

        return indexes

    def str2tensor(self, strings):
        """Convert text-string to input tensor.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            tensors (list[torch.Tensor]): [torch.Tensor([1,2,3,3,4]),
                torch.Tensor([5,4,6,3,7])].
        """
        raise NotImplementedError

    def idx2str(self, indexes):
        """Convert indexes to text strings.

        Args:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        Returns:
            strings (list[str]): ['hello', 'world'].
        """
        assert isinstance(indexes, list)

        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            strings.append(''.join(string))

        return strings

    def tensor2idx(self, output):
        """Convert model output tensor to character indexes and scores.
        Args:
            output (tensor): The model outputs with size: N * T * C
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                [0.9,0.9,0.98,0.97,0.96]].
        """
        raise NotImplementedError
