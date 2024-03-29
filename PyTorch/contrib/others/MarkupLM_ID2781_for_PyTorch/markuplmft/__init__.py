# ============================================================================
# Copyright 2022 Huawei Technologies Co., Ltd
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

from transformers import CONFIG_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING, \
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter
from transformers.file_utils import PRESET_MIRROR_DICT

from .models.markuplm import (
    MarkupLMConfig,
    MarkupLMTokenizer,
    MarkupLMForQuestionAnswering,
    MarkupLMForTokenClassification,
    MarkupLMTokenizerFast,
)

CONFIG_MAPPING.update(
    [
        ("markuplm", MarkupLMConfig),
    ]
)
MODEL_NAMES_MAPPING.update([("markuplm", "MarkupLM")])

TOKENIZER_MAPPING.update(
    [
        (MarkupLMConfig, (MarkupLMTokenizer, MarkupLMTokenizerFast)),
    ]
)

SLOW_TO_FAST_CONVERTERS.update(
    {"MarkupLMTokenizer": RobertaConverter}
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING.update(
    [(MarkupLMConfig, MarkupLMForQuestionAnswering)]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [(MarkupLMConfig, MarkupLMForTokenClassification)]
)
