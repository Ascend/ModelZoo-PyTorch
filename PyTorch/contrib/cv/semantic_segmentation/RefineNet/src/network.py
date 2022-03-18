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
# ============================================================================
import logging
import re
from models.refinenet import rf101


def get_segmenter(
    enc_backbone, enc_pretrained, num_classes,
):
    """Create Encoder-Decoder; for now only ResNet [101] Encoders are supported"""
    if enc_backbone == "101":
        return rf101(num_classes, imagenet=enc_pretrained)
    else:
        raise ValueError("{} is not supported".format(str(enc_backbone)))


def get_encoder_and_decoder_params(model):
    """Filter model parameters into two groups: encoder and decoder."""
    logger = logging.getLogger(__name__)
    enc_params = []
    dec_params = []
    for k, v in model.named_parameters():
        if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
            enc_params.append(v)
            logger.info(" Enc. parameter: {}".format(k))
            # print((" Enc. parameter: {}".format(k)))
        else:
            dec_params.append(v)
            logger.info(" Dec. parameter: {}".format(k))
            # print((" Dec. parameter: {}".format(k)))
    return enc_params, dec_params
