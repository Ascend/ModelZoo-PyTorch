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
import functools
import numpy as np
from espnet_onnx.asr.model.encoder import get_encoder
from espnet_onnx.asr.model.decoder import get_decoder
from espnet_onnx.asr.model.lm import get_lm
from espnet_onnx.asr.model.joint_network import JointNetwork
from espnet_onnx.asr.scorer.ctc_prefix_scorer import CTCPrefixScorer
from espnet_onnx.asr.scorer.length_bonus import LengthBonus


####################################
# Added function for npu inference #
####################################
def asr_npu_call(self, speech: np.ndarray):
    """
    NPU inference for ASR model
    """
    if not self.only_use_decoder:
        if isinstance(speech, np.ndarray):
            speech = speech[np.newaxis, :]
            lengths = np.array([speech.shape[1]]).astype("int64")
            enc, _ = self.encoder(speech=speech, speech_length=lengths)
        else:
            # support for multibatch
            enc, _ = self.encoder(speech, None)
    else:
        enc = [speech]

    if self.only_use_encoder:
        return enc

    enc = enc[0]
    init_batch = enc.shape[0]
    nbest_hyps_list = self.beam_search(enc)
    nbest_hyps = []
    for batch_idx in range(init_batch):
        nbest_hyps.append(nbest_hyps_list[batch_idx][0])
    return nbest_hyps


def build_speech_model_npu(self, device_id, only_use_encoder, only_use_decoder, rank_mode,
                           disable_preprocess=True, enable_multibatch=True, use_quantized=False):
    """
    Build NPU models for ASR model
    """
    self.device_id = device_id
    self.only_use_encoder = only_use_encoder
    self.only_use_decoder = only_use_decoder
    self.enable_multibatch = enable_multibatch

    if not only_use_decoder:
        self.encoder = get_encoder(
            self.config.encoder, self.providers, use_quantized, use_npu=True,
            rank_mode=rank_mode, device_id=device_id, disable_preprocess=disable_preprocess,
            fp16=False)
        if only_use_encoder:
            return

    decoder = get_decoder(self.config.decoder, self.providers, use_quantized,
                          use_npu=True, device_id=device_id, fp16=True)
    scorers = {'decoder': decoder}
    weights = {}
    if not self.config.transducer.use_transducer_decoder:
        use_ctc = True
        if 'use_ctc' in self.config.ctc.keys():
            use_ctc = self.config.ctc.use_ctc
        if use_ctc:
            ctc = CTCPrefixScorer(self.config.ctc, self.config.token.eos, self.providers,
                                  use_quantized, device_id=device_id, fp16=True)
            scorers.update(
                ctc=ctc,
                length_bonus=LengthBonus(len(self.config.token.list))
            )
            weights.update(
                decoder=self.config.weights.decoder,

                ctc=self.config.weights.ctc,
                length_bonus=self.config.weights.length_bonus,
            )
        else:
            weights.update(
                decoder=self.config.weights.decoder,
            )
    else:
        joint_network = JointNetwork(self.config.joint_network, self.providers, use_quantized,
                                     use_npu=True, device_id=device_id)
        scorers.update(joint_network=joint_network)

    lm = get_lm(self.config, self.providers, use_quantized,
                use_npu=True, device_id=device_id, fp16=True)
    if lm is not None:
        scorers.update(lm=lm)
        weights.update(lm=self.config.weights.lm)

    self._build_beam_search(scorers, weights)
    self._build_tokenizer()
    self._build_token_converter()
    self.scorers = scorers
    self.weights = weights


#################################
# Decorator for NPU model class #
#################################
def init_speech_npu(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        providers = kwargs['providers']
        if providers[0] == 'NPUExecutionProvider':
            self = args[0]
            model_dir = kwargs['model_dir']
            self._check_argument(None, model_dir)
            self._load_config()
            self.providers = providers
            self.build_speech_model_npu(
                device_id=kwargs['device_id'],
                only_use_encoder=kwargs['only_use_encoder'],
                only_use_decoder=kwargs['only_use_decoder'],
                disable_preprocess=kwargs['disable_preprocess'],
                enable_multibatch=kwargs['enable_multibatch'],
                rank_mode=kwargs['rank_mode']
            )
            if self.config.transducer.use_transducer_decoder:
                self.start_idx = 1
                self.last_idx = None
            else:
                self.start_idx = 1
                self.last_idx = -1
        else:
            func(*args, **kwargs)
    return wrapper


def speech_npu_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.providers[0] == 'NPUExecutionProvider':
            speech = args[1]
            nbest_hyps = self.asr_npu_call(speech)
            if self.only_use_encoder:
                return nbest_hyps
            else:
                return self._post_process(nbest_hyps)
        else:
            return func(*args, **kwargs)
    return wrapper


def speech_npu_adapt(cls):
    setattr(cls, 'build_speech_model_npu', build_speech_model_npu)
    setattr(cls, 'asr_npu_call', asr_npu_call)
    return cls
