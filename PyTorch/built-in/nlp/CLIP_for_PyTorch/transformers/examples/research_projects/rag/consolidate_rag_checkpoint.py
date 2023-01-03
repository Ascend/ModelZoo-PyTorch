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

"""
A script creating a RAG checkpoint from a generator and a question encoder checkpoints.
"""

import argparse
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, RagConfig, RagSequenceForGeneration, RagTokenForGeneration


def consolidate(
    model_type,
    generator_name_or_path: str,
    question_encoder_name_or_path: str,
    dest_dir: Path,
    config_name_or_path: str = None,
    generator_tokenizer_name_or_path: str = None,
    question_encoder_tokenizer_name_or_path: str = None,
):

    if config_name_or_path is None:
        config_name_or_path = "facebook/rag-token-base" if model_type == "rag_token" else "facebook/rag-sequence-base"

    if generator_tokenizer_name_or_path is None:
        generator_tokenizer_name_or_path = generator_name_or_path

    if question_encoder_tokenizer_name_or_path is None:
        question_encoder_tokenizer_name_or_path = question_encoder_name_or_path

    model_class = RagTokenForGeneration if model_type == "rag_token" else RagSequenceForGeneration

    # Save model.
    rag_config = RagConfig.from_pretrained(config_name_or_path)
    gen_config = AutoConfig.from_pretrained(generator_name_or_path)
    question_encoder_config = AutoConfig.from_pretrained(question_encoder_name_or_path)

    rag_config.generator = gen_config
    rag_config.question_encoder = question_encoder_config

    rag_model = model_class.from_pretrained_question_encoder_generator(
        question_encoder_name_or_path, generator_name_or_path, config=rag_config
    )
    rag_model.save_pretrained(dest_dir)

    # Sanity check.
    model_class.from_pretrained(dest_dir)

    # Save tokenizers.
    gen_tokenizer = AutoTokenizer.from_pretrained(generator_tokenizer_name_or_path)
    gen_tokenizer.save_pretrained(dest_dir / "generator_tokenizer/")
    question_encoder_tokenizer = AutoTokenizer.from_pretrained(question_encoder_tokenizer_name_or_path)
    question_encoder_tokenizer.save_pretrained(dest_dir / "question_encoder_tokenizer/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["rag_sequence", "rag_token"],
        required=True,
        type=str,
        help="RAG model type: rag_sequence, rag_token",
    )
    parser.add_argument("--dest", type=str, required=True, help="Path to the output checkpoint directory.")
    parser.add_argument("--generator_name_or_path", type=str, required=True, help="Generator model identifier")
    parser.add_argument(
        "--question_encoder_name_or_path", type=str, required=True, help="Question encoder model identifier"
    )

    parser.add_argument(
        "--generator_tokenizer_name_or_path",
        type=str,
        help="Generator tokenizer identifier, if not specified, resolves to ``generator_name_or_path``",
    )
    parser.add_argument(
        "--question_encoder_tokenizer_name_or_path",
        type=str,
        help="Question encoder tokenizer identifier, if not specified, resolves to ``question_encoder_name_or_path``",
    )
    parser.add_argument(
        "--config_name_or_path",
        type=str,
        help="Identifier of the model config to use, if not provided, resolves to a base config for a given ``model_type``",
    )

    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(exist_ok=True)

    consolidate(
        args.model_type,
        args.generator_name_or_path,
        args.question_encoder_name_or_path,
        dest_dir,
        args.config_name_or_path,
        args.generator_tokenizer_name_or_path,
        args.question_encoder_tokenizer_name_or_path,
    )
