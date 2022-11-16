# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_rembert": ["REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RemBertConfig"],
}

if is_sentencepiece_available():
    _import_structure["tokenization_rembert"] = ["RemBertTokenizer"]

if is_tokenizers_available():
    _import_structure["tokenization_rembert_fast"] = ["RemBertTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_rembert"] = [
        "REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RemBertForCausalLM",
        "RemBertForMaskedLM",
        "RemBertForMultipleChoice",
        "RemBertForQuestionAnswering",
        "RemBertForSequenceClassification",
        "RemBertForTokenClassification",
        "RemBertLayer",
        "RemBertModel",
        "RemBertPreTrainedModel",
        "load_tf_weights_in_rembert",
    ]


if is_tf_available():
    _import_structure["modeling_tf_rembert"] = [
        "TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFRemBertForCausalLM",
        "TFRemBertForMaskedLM",
        "TFRemBertForMultipleChoice",
        "TFRemBertForQuestionAnswering",
        "TFRemBertForSequenceClassification",
        "TFRemBertForTokenClassification",
        "TFRemBertLayer",
        "TFRemBertModel",
        "TFRemBertPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_rembert import REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RemBertConfig

    if is_sentencepiece_available():
        from .tokenization_rembert import RemBertTokenizer

    if is_tokenizers_available():
        from .tokenization_rembert_fast import RemBertTokenizerFast

    if is_torch_available():
        from .modeling_rembert import (
            REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            RemBertForCausalLM,
            RemBertForMaskedLM,
            RemBertForMultipleChoice,
            RemBertForQuestionAnswering,
            RemBertForSequenceClassification,
            RemBertForTokenClassification,
            RemBertLayer,
            RemBertModel,
            RemBertPreTrainedModel,
            load_tf_weights_in_rembert,
        )

    if is_tf_available():
        from .modeling_tf_rembert import (
            TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRemBertForCausalLM,
            TFRemBertForMaskedLM,
            TFRemBertForMultipleChoice,
            TFRemBertForQuestionAnswering,
            TFRemBertForSequenceClassification,
            TFRemBertForTokenClassification,
            TFRemBertLayer,
            TFRemBertModel,
            TFRemBertPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)