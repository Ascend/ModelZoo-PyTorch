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

from src.utils import loggers
import src.config.constants as constants
from src.utils.configs import Configuration
import traceback
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import sequence_padding
import numpy as np

class BertModel():
    def __init__(self):
        rec_info_config = Configuration.configurations.get(constants.CONFIG_MODEL_KEY)
        self.model_path = rec_info_config.get(constants.CONFIG_MODEL_PATH)
        self.vocab_path = rec_info_config.get(constants.CONFIG_MODEL_VOCAB)
        self.mapping = {0: 'negative', 1: 'positive'}

    def load_model(self):
        try:
            import onnxruntime
            self.model = onnxruntime.InferenceSession(self.model_path)
            self.tokenizer = Tokenizer(self.vocab_path, do_lower_case=True)
        except Exception as ex:
            loggers.get_error_log().error("An exception occured while load model: {}".format(traceback.format_exc()))

    async def process(self, user_inputs):
        user_inputs = [user_inputs] if isinstance(user_inputs, str) else user_inputs
        input_ids, segment_ids = self.tokenizer.encode(user_inputs)
        input_ids = sequence_padding(input_ids).astype('int64')
        segment_ids = sequence_padding(segment_ids).astype('int64')

        # 模型推理结果
        ort_inputs = {self.model.get_inputs()[0].name: input_ids,
                    self.model.get_inputs()[1].name: segment_ids}
        ort_outs = self.model.run(None, ort_inputs)
        ort_outs = list(np.argmax(ort_outs[0], axis=1))
        return [{k:v} for k, v in zip(user_inputs, [self.mapping[i] for i in ort_outs])]