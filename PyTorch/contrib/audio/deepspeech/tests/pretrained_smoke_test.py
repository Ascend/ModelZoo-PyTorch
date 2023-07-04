# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import unittest

import wget

from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.enums import DecoderType
from tests.smoke_test import DatasetConfig, DeepSpeechSmokeTest
with open('../url.ini', 'r') as f:
    content = f.read()
    an4_pretrained_v2 = content.split('an4_pretrained_v2=')[1].split('\n')[0]
    librispeech_pretrained_v2 = content.split('librispeech_pretrained_v2=')[1].split('\n')[0]
    ted_pretrained_v2 = content.split('ted_pretrained_v2=')[1].split('\n')[0]
pretrained_urls = [
    an4_pretrained_v2,
    librispeech_pretrained_v2,
    ted_pretrained_v2
]

lm_path = 'http://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz'


class PretrainedSmokeTest(DeepSpeechSmokeTest):

    def test_pretrained_eval_inference(self):
        # Disabled GPU due to using TravisCI
        cuda, use_half = False, False
        train_manifest, val_manifest, test_manifest = self.download_data(DatasetConfig(target_dir=self.target_dir,
                                                                                       manifest_dir=self.manifest_dir))
        wget.download(lm_path)
        for pretrained_url in pretrained_urls:
            print("Running Pre-trained Smoke test for: ", pretrained_url)
            wget.download(pretrained_url)
            file_path = os.path.basename(pretrained_url)
            pretrained_path = os.path.abspath(file_path)

            lm_configs = [
                LMConfig(),  # Greedy
                LMConfig(
                    decoder_type=DecoderType.beam
                ),  # Test Beam Decoder
                LMConfig(
                    decoder_type=DecoderType.beam,
                    lm_path=os.path.basename(lm_path),
                    alpha=1,
                    beta=1
                )  # Test Beam Decoder with LM
            ]

            for lm_config in lm_configs:
                self.eval_model(model_path=pretrained_path,
                                test_manifest=test_manifest,
                                cuda=cuda,
                                use_half=use_half,
                                lm_config=lm_config)
                self.inference(test_manifest=test_manifest,
                               model_path=pretrained_path,
                               cuda=cuda,
                               lm_config=lm_config,
                               use_half=use_half)


if __name__ == '__main__':
    unittest.main()
