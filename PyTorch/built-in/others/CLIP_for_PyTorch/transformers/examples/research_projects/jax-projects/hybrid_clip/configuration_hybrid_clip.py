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

import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class HybridCLIPConfig(PretrainedConfig):
    r"""
    :class:`HybridCLIPConfig` is the configuration class to store the configuration of a
    :class:`~HybridCLIPModel`. It is used to instantiate HybridCLIPModel model according to the specified arguments,
    defining the text model and vision model configs.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        text_config_dict (:obj:`dict`):
            Dictionary of configuration options that defines text model config.
        vision_config_dict (:obj:`dict`):
            Dictionary of configuration options that defines vison model config.
        projection_dim (:obj:`int`, `optional`, defaults to 512):
            Dimentionality of text and vision projection layers.
        kwargs (`optional`):
            Dictionary of keyword arguments.

    Examples::

        >>> from transformers import BertConfig, CLIPConfig, HybridCLIPConfig, FlaxHybridCLIP

        >>> # Initializing a BERT and CLIP configuration
        >>> config_text = BertConfig()
        >>> config_vision = CLIPConfig()

        >>> config = HybridCLIPConfig.from_text_vision_configs(config_text, config_vision, projection_dim=512)

        >>> # Initializing a BERT and CLIPVision model
        >>> model = EncoderDecoderModel(config=config)

        >>> # Accessing the model configuration
        >>> config_text = model.config.text_config
        >>> config_vision  = model.config.vision_config

        >>> # Saving the model, including its configuration
        >>> model.save_pretrained('my-model')

        >>> # loading model and config from pretrained folder
        >>> encoder_decoder_config = HybridCLIPConfig.from_pretrained('my-model')
        >>> model = FlaxHybridCLIP.from_pretrained('my-model', config=encoder_decoder_config)
    """

    model_type = "hybrid-clip"
    is_composition = True

    def __init__(self, projection_dim=512, **kwargs):
        super().__init__(**kwargs)

        if "text_config" not in kwargs:
            raise ValueError("`text_config` can not be `None`.")

        if "vision_config" not in kwargs:
            raise ValueError("`vision_config` can not be `None`.")

        text_config = kwargs.pop("text_config")
        vision_config = kwargs.pop("vision_config")

        text_model_type = text_config.pop("model_type")
        vision_model_type = vision_config.pop("model_type")

        from transformers import AutoConfig

        self.text_config = AutoConfig.for_model(text_model_type, **text_config)

        if vision_model_type == "clip":
            self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config).vision_config
        elif vision_model_type == "clip_vision_model":
            from transformers import CLIPVisionConfig

            self.vision_config = CLIPVisionConfig(**vision_config)
        else:
            self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config)

        self.projection_dim = projection_dim
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: PretrainedConfig, vision_config: PretrainedConfig, **kwargs):
        r"""
        Instantiate a :class:`HybridCLIPConfig` (or a derived class) from text model configuration and
        vision model configuration.

        Returns:
            :class:`HybridCLIPConfig`: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default
        :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
