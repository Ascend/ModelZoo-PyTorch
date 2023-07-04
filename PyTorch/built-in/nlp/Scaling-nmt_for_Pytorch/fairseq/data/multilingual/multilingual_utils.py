# coding:utf-8
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


from enum import Enum
from typing import Dict, List, Optional, Sequence

import torch
from fairseq.data import Dictionary


class EncoderLangtok(Enum):
    """
    Prepend to the beginning of source sentence either the
    source or target language token. (src/tgt).
    """

    src = "src"
    tgt = "tgt"


class LangTokSpec(Enum):
    main = "main"
    mono_dae = "mono_dae"


class LangTokStyle(Enum):
    multilingual = "multilingual"
    mbart = "mbart"


@torch.jit.export
def get_lang_tok(
    lang: str, lang_tok_style: str, spec: str = LangTokSpec.main.value
) -> str:
    # TOKEN_STYLES can't be defined outside this fn since it needs to be
    # TorchScriptable.
    TOKEN_STYLES: Dict[str, str] = {
        LangTokStyle.mbart.value: "[{}]",
        LangTokStyle.multilingual.value: "__{}__",
    }

    if spec.endswith("dae"):
        lang = f"{lang}_dae"
    elif spec.endswith("mined"):
        lang = f"{lang}_mined"
    style = TOKEN_STYLES[lang_tok_style]
    return style.format(lang)


def augment_dictionary(
    dictionary: Dictionary,
    language_list: List[str],
    lang_tok_style: str,
    langtoks_specs: Sequence[str] = (LangTokSpec.main.value,),
    extra_data: Optional[Dict[str, str]] = None,
) -> None:
    for spec in langtoks_specs:
        for language in language_list:
            dictionary.add_symbol(
                get_lang_tok(lang=language, lang_tok_style=lang_tok_style, spec=spec)
            )

    if lang_tok_style == LangTokStyle.mbart.value or (
        extra_data is not None and LangTokSpec.mono_dae.value in extra_data
    ):
        dictionary.add_symbol("<mask>")
