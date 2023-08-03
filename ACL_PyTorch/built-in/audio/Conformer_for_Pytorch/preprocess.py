# Copyright 2023 Huawei Technologies Co., Ltd
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


from typing import (
    Tuple,
    List
)
import numpy as np

from espnet_onnx.asr.frontend.frontend import Frontend
from espnet_onnx.asr.frontend.normalize.global_mvn import GlobalMVN
from espnet_onnx.asr.frontend.normalize.utterance_mvn import UtteranceMVN
from espnet_onnx.utils.config import get_config


class Preprocessor:
    def __init__(
            self,
            config_path: str,
            rank_nums = None,
            providers = ['NPUExecutionProvider'],
            rank_mode: bool = False,
            use_quantized: bool = False):
        self.config = get_config(config_path).encoder
        self.rank_mode = rank_mode
        self.rank_nums = np.array(rank_nums)
        self.frontend = Frontend(self.config.frontend, providers, use_quantized)
        if self.config.do_normalize:
            if self.config.normalize.type == 'gmvn':
                self.normalize = GlobalMVN(self.config.normalize)
            elif self.config.normalize.type == 'utterance_mvn':
                self.normalize = UtteranceMVN(self.config.normalize)

    def _build_mask(self, input_data, input_lens, max_len=None, hidden_size=1):
        """
        Build seq_mask/pos_mask/conv_mask for rank mode
        """
        seq_len = (input_lens - 3) // 2 + 1
        seq_len = (seq_len - 3) // 2 + 1
        batch = input_data.shape[0]
        if max_len is None:
            max_len = max(seq_len)
        else:
            max_len = (max_len - 3) // 2 + 1
            max_len = (max_len - 3) // 2 + 1
        # build mask
        mask = np.ones([batch, max_len]).astype("float32")
        conv_mask = np.ones([batch, 1, max_len]).astype("float32")
        encoder_out_lens = np.ones([batch, 1, max_len])
        for idx in range(batch):
            mask[idx, np.arange(seq_len[idx], mask.shape[1])] = 0
            conv_mask[idx, :, np.arange(seq_len[idx], mask.shape[1])] = 0
            encoder_out_lens[idx, :, np.arange(seq_len[idx], mask.shape[1])] = 1

        # build relative pos mask
        def generate_relative_pos(seq):
            length = seq * (seq + 1)
            mask = np.arange(length).reshape(seq, seq+1)
            mask = mask.reshape([seq + 1, seq])
            mask = mask[1:]
            # generate offset for padded cases
            pad_size = max_len - seq
            if not pad_size:
                mask = np.expand_dims(mask, axis=[0])
                mask = np.tile(mask, (hidden_size, 1, 1))
                return mask.reshape([hidden_size,  -1])
            mask_size = mask.shape[0] * mask.shape[1]
            stride_num = round((mask_size - 1) / (seq + 1))
            offset = []
            for stride in range(pad_size, stride_num*pad_size + 1, pad_size):
                offset.append([stride] * (seq + 1))
            offset = np.array(offset)
            offset = np.array([0] + offset.reshape(-1).tolist()).reshape(mask.shape)
            mask += offset
            mask = np.expand_dims(np.pad(mask, ((0, max_len-seq), (0, max_len-seq))), axis=[0])
            mask = np.tile(mask, (hidden_size, 1, 1))
            return mask.reshape([hidden_size, -1])

        pos_mask = np.array([generate_relative_pos(seq) for seq in seq_len]).astype("float32")
        pos_mask = pos_mask.reshape([batch * hidden_size, -1])
        return mask, conv_mask, pos_mask, encoder_out_lens.astype("int64")

    def __call__(self, speech):
        if isinstance(speech, np.ndarray):
            speech = speech[np.newaxis, :]
            speech_length = np.array([speech.shape[1]]).astype("int64")

            # 1. Extract feature
            feats, feat_length = self.frontend(speech, speech_length)

            # 2. normalize with global MVN
            if self.config.do_normalize:
                feats, feat_length = self.normalize(feats, feat_length)
        else:
            speech_length = np.array([_.shape[0] for _ in speech]).astype("int64")
            feats_list = []
            feat_lengths = []
            seq_lens = []
            for idx, sp in enumerate(speech):
                # 1. Extract feature
                sp = sp[np.newaxis, :]
                le = np.array([speech_length[idx]])
                feats, feat_length = self.frontend(sp, le)

                # 2. normalize with global MVN
                if self.config.do_normalize:
                    feats, feat_length = self.normalize(feats, feat_length)

                feats_list.append(feats)
                feat_lengths.append(feat_length[0])
                seq_lens.append(feats.shape[1])
            # 3. pad feats
            max_len = max([_.shape[1] for _ in feats_list])
            if self.rank_mode:
                max_len = self.rank_nums[self.rank_nums >= max_len][0]

            feats = np.concatenate([np.pad(_, ((0, 0), (0, max_len - _.shape[1]), (0, 0))) for _ in feats_list], axis=0)
            feat_lengths = np.array(feat_lengths)
            seq_lens = np.array(seq_lens)
            mask, conv_mask, pos_mask, encoder_out_lens = self._build_mask(feats, seq_lens, max_len=feats.shape[1])
            return (feats, feat_lengths, mask, pos_mask, conv_mask, encoder_out_lens)
