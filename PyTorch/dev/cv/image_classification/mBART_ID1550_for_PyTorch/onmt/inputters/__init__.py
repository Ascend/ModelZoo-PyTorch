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
"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from onmt.inputters.inputter import \
    load_old_vocab, get_fields, OrderedIterator, \
    build_vocab, old_style_vocab, filter_example
from onmt.inputters.dataset_base import Dataset
from onmt.inputters.text_dataset import text_sort_key, TextDataReader
from onmt.inputters.image_dataset import img_sort_key, ImageDataReader
from onmt.inputters.audio_dataset import audio_sort_key, AudioDataReader
from onmt.inputters.vec_dataset import vec_sort_key, VecDataReader
from onmt.inputters.datareader_base import DataReaderBase
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))

str2reader = {
    "text": TextDataReader, "img": ImageDataReader, "audio": AudioDataReader,
    "vec": VecDataReader}
str2sortkey = {
    'text': text_sort_key, 'img': img_sort_key, 'audio': audio_sort_key,
    'vec': vec_sort_key}


__all__ = ['Dataset', 'load_old_vocab', 'get_fields', 'DataReaderBase',
           'filter_example', 'old_style_vocab',
           'build_vocab', 'OrderedIterator',
           'text_sort_key', 'img_sort_key', 'audio_sort_key', 'vec_sort_key',
           'TextDataReader', 'ImageDataReader', 'AudioDataReader',
           'VecDataReader']
