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

from .dictionary import Dictionary, TruncatedDictionary

from .fairseq_dataset import FairseqDataset, FairseqIterableDataset

from .base_wrapper_dataset import BaseWrapperDataset

from .append_token_dataset import AppendTokenDataset
from .audio.raw_audio_dataset import FileAudioDataset
from .backtranslation_dataset import BacktranslationDataset
from .colorize_dataset import ColorizeDataset
from .concat_dataset import ConcatDataset
from .concat_sentences_dataset import ConcatSentencesDataset
from .denoising_dataset import DenoisingDataset
from .id_dataset import IdDataset
from .indexed_dataset import IndexedCachedDataset, IndexedDataset, IndexedRawTextDataset, MMapIndexedDataset
from .language_pair_dataset import LanguagePairDataset
from .list_dataset import ListDataset
from .lm_context_window_dataset import LMContextWindowDataset
from .lru_cache_dataset import LRUCacheDataset
from .mask_tokens_dataset import MaskTokensDataset
from .monolingual_dataset import MonolingualDataset
from .nested_dictionary_dataset import NestedDictionaryDataset
from .noising import NoisingDataset
from .numel_dataset import NumelDataset
from .num_samples_dataset import NumSamplesDataset
from .offset_tokens_dataset import OffsetTokensDataset
from .pad_dataset import LeftPadDataset, PadDataset, RightPadDataset
from .prepend_dataset import PrependDataset
from .prepend_token_dataset import PrependTokenDataset
from .raw_label_dataset import RawLabelDataset
from .replace_dataset import ReplaceDataset
from .resampling_dataset import ResamplingDataset
from .roll_dataset import RollDataset
from .round_robin_zip_datasets import RoundRobinZipDatasets
from .sharded_dataset import ShardedDataset
from .sort_dataset import SortDataset
from .strip_token_dataset import StripTokenDataset
from .subsample_dataset import SubsampleDataset
from .token_block_dataset import TokenBlockDataset
from .transform_eos_dataset import TransformEosDataset
from .transform_eos_lang_pair_dataset import TransformEosLangPairDataset
from .truncate_dataset import TruncateDataset

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
)

__all__ = [
    'AppendTokenDataset',
    'BacktranslationDataset',
    'BaseWrapperDataset',
    'ColorizeDataset',
    'ConcatDataset',
    'ConcatSentencesDataset',
    'CountingIterator',
    'DenoisingDataset',
    'Dictionary',
    'EpochBatchIterator',
    'FairseqDataset',
    'FairseqIterableDataset',
    'GroupedIterator',
    'IdDataset',
    'IndexedCachedDataset',
    'IndexedDataset',
    'IndexedRawTextDataset',
    'LanguagePairDataset',
    'LeftPadDataset',
    'ListDataset',
    'LMContextWindowDataset',
    'LRUCacheDataset',
    'MaskTokensDataset',
    'MMapIndexedDataset',
    'MonolingualDataset',
    'NestedDictionaryDataset',
    'NoisingDataset',
    'NumelDataset',
    'NumSamplesDataset',
    'OffsetTokensDataset',
    'PadDataset',
    'PrependDataset',
    'PrependTokenDataset',
    'ReplaceDataset',
    'RollDataset',
    'FileAudioDataset',
    'RawLabelDataset',
    'ResamplingDataset',
    'RightPadDataset',
    'RoundRobinZipDatasets',
    'ShardedDataset',
    'ShardedIterator',
    'SortDataset',
    'StripTokenDataset',
    'SubsampleDataset',
    'TokenBlockDataset',
    'TransformEosDataset',
    'TransformEosLangPairDataset',
    'TruncateDataset',
    'TruncatedDictionary',
]
