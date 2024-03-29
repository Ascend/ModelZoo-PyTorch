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
    is_speech_available,
    is_tf_available,
    is_torch_available,
)


_import_structure = {
    "configuration_speech_to_text": [
        "SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Speech2TextConfig",
    ],
}

if is_sentencepiece_available():
    _import_structure["tokenization_speech_to_text"] = ["Speech2TextTokenizer"]

if is_speech_available():
    _import_structure["feature_extraction_speech_to_text"] = ["Speech2TextFeatureExtractor"]

    if is_sentencepiece_available():
        _import_structure["processing_speech_to_text"] = ["Speech2TextProcessor"]

if is_tf_available():
    _import_structure["modeling_tf_speech_to_text"] = [
        "TF_SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSpeech2TextForConditionalGeneration",
        "TFSpeech2TextModel",
        "TFSpeech2TextPreTrainedModel",
    ]

if is_torch_available():
    _import_structure["modeling_speech_to_text"] = [
        "SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Speech2TextForConditionalGeneration",
        "Speech2TextModel",
        "Speech2TextPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_speech_to_text import SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP, Speech2TextConfig

    if is_sentencepiece_available():
        from .tokenization_speech_to_text import Speech2TextTokenizer

    if is_speech_available():
        from .feature_extraction_speech_to_text import Speech2TextFeatureExtractor

        if is_sentencepiece_available():
            from .processing_speech_to_text import Speech2TextProcessor

    if is_tf_available():
        from .modeling_tf_speech_to_text import (
            TF_SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSpeech2TextForConditionalGeneration,
            TFSpeech2TextModel,
            TFSpeech2TextPreTrainedModel,
        )

    if is_torch_available():
        from .modeling_speech_to_text import (
            SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            Speech2TextForConditionalGeneration,
            Speech2TextModel,
            Speech2TextPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
