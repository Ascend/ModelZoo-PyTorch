#!/usr/bin/env python3
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

# import models/encoder/decoder to be tested
from examples.speech_recognition.models.vggtransformer import (
    TransformerDecoder,
    VGGTransformerEncoder,
    VGGTransformerModel,
    vggtransformer_1,
    vggtransformer_2,
    vggtransformer_base,
)

# import base test class
from .asr_test_base import (
    DEFAULT_TEST_VOCAB_SIZE,
    TestFairseqDecoderBase,
    TestFairseqEncoderBase,
    TestFairseqEncoderDecoderModelBase,
    get_dummy_dictionary,
    get_dummy_encoder_output,
    get_dummy_input,
)


class VGGTransformerModelTest_mid(TestFairseqEncoderDecoderModelBase):
    def setUp(self):
        def override_config(args):
            """
            vggtrasformer_1 use 14 layers of transformer,
            for testing purpose, it is too expensive. For fast turn-around
            test, reduce the number of layers to 3.
            """
            args.transformer_enc_config = (
                "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 3"
            )

        super().setUp()
        extra_args_setter = [vggtransformer_1, override_config]

        self.setUpModel(VGGTransformerModel, extra_args_setter)
        self.setUpInput(get_dummy_input(T=50, D=80, B=5, K=DEFAULT_TEST_VOCAB_SIZE))


class VGGTransformerModelTest_big(TestFairseqEncoderDecoderModelBase):
    def setUp(self):
        def override_config(args):
            """
            vggtrasformer_2 use 16 layers of transformer,
            for testing purpose, it is too expensive. For fast turn-around
            test, reduce the number of layers to 3.
            """
            args.transformer_enc_config = (
                "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 3"
            )

        super().setUp()
        extra_args_setter = [vggtransformer_2, override_config]

        self.setUpModel(VGGTransformerModel, extra_args_setter)
        self.setUpInput(get_dummy_input(T=50, D=80, B=5, K=DEFAULT_TEST_VOCAB_SIZE))


class VGGTransformerModelTest_base(TestFairseqEncoderDecoderModelBase):
    def setUp(self):
        def override_config(args):
            """
            vggtrasformer_base use 12 layers of transformer,
            for testing purpose, it is too expensive. For fast turn-around
            test, reduce the number of layers to 3.
            """
            args.transformer_enc_config = (
                "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 3"
            )

        super().setUp()
        extra_args_setter = [vggtransformer_base, override_config]

        self.setUpModel(VGGTransformerModel, extra_args_setter)
        self.setUpInput(get_dummy_input(T=50, D=80, B=5, K=DEFAULT_TEST_VOCAB_SIZE))


class VGGTransformerEncoderTest(TestFairseqEncoderBase):
    def setUp(self):
        super().setUp()

        self.setUpInput(get_dummy_input(T=50, D=80, B=5))

    def test_forward(self):
        print("1. test standard vggtransformer")
        self.setUpEncoder(VGGTransformerEncoder(input_feat_per_channel=80))
        super().test_forward()
        print("2. test vggtransformer with limited right context")
        self.setUpEncoder(
            VGGTransformerEncoder(
                input_feat_per_channel=80, transformer_context=(-1, 5)
            )
        )
        super().test_forward()
        print("3. test vggtransformer with limited left context")
        self.setUpEncoder(
            VGGTransformerEncoder(
                input_feat_per_channel=80, transformer_context=(5, -1)
            )
        )
        super().test_forward()
        print("4. test vggtransformer with limited right context and sampling")
        self.setUpEncoder(
            VGGTransformerEncoder(
                input_feat_per_channel=80,
                transformer_context=(-1, 12),
                transformer_sampling=(2, 2),
            )
        )
        super().test_forward()
        print("5. test vggtransformer with windowed context and sampling")
        self.setUpEncoder(
            VGGTransformerEncoder(
                input_feat_per_channel=80,
                transformer_context=(12, 12),
                transformer_sampling=(2, 2),
            )
        )


class TransformerDecoderTest(TestFairseqDecoderBase):
    def setUp(self):
        super().setUp()

        dict = get_dummy_dictionary(vocab_size=DEFAULT_TEST_VOCAB_SIZE)
        decoder = TransformerDecoder(dict)
        dummy_encoder_output = get_dummy_encoder_output(encoder_out_shape=(50, 5, 256))

        self.setUpDecoder(decoder)
        self.setUpInput(dummy_encoder_output)
        self.setUpPrevOutputTokens()
