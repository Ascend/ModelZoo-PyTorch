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

from dataclasses import dataclass

from deepspeech_pytorch.enums import DecoderType


@dataclass
class LMConfig:
    decoder_type: DecoderType = DecoderType.greedy
    lm_path: str = ''  # Path to an (optional) kenlm language model for use with beam search (req\'d with trie)
    top_paths: int = 1  # Number of beams to return
    alpha: float = 0.0  # Language model weight
    beta: float = 0.0  # Language model word bonus (all words)
    cutoff_top_n: int = 40  # Cutoff_top_n characters with highest probs in vocabulary will be used in beam search
    cutoff_prob: float = 1.0  # Cutoff probability in pruning,default 1.0, no pruning.
    beam_width: int = 10  # Beam width to use
    lm_workers: int = 4  # Number of LM processes to use


@dataclass
class ModelConfig:
    use_half: bool = True  # Use half precision. This is recommended when using mixed-precision at training time
    cuda: bool = True
    model_path: str = ''


@dataclass
class InferenceConfig:
    lm: LMConfig = LMConfig()
    model: ModelConfig = ModelConfig()


@dataclass
class TranscribeConfig(InferenceConfig):
    audio_path: str = ''  # Audio file to predict on
    offsets: bool = False  # Returns time offset information


@dataclass
class EvalConfig(InferenceConfig):
    test_manifest: str = ''  # Path to validation manifest csv
    verbose: bool = True  # Print out decoded output and error of each sample
    save_output: str = ''  # Saves output of model from test to this file_path
    batch_size: int = 20  # Batch size for testing
    num_workers: int = 4


@dataclass
class ServerConfig(InferenceConfig):
    host: str = '0.0.0.0'
    port: int = 8888
