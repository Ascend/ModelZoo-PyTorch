#
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
#
import os

from .parser_image_folder import ParserImageFolder
from .parser_image_tar import ParserImageTar
from .parser_image_in_tar import ParserImageInTar
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


def create_parser(name, root, split='train', **kwargs):
    name = name.lower()
    name = name.split('/', 2)
    prefix = ''
    if len(name) > 1:
        prefix = name[0]
    name = name[-1]

    # FIXME improve the selection right now just tfds prefix or fallback path, will need options to
    # explicitly select other options shortly
    if prefix == 'tfds':
        from .parser_tfds import ParserTfds  # defer tensorflow import
        parser = ParserTfds(root, name, split=split, shuffle=kwargs.pop('shuffle', False), **kwargs)
    else:
        assert os.path.exists(root)
        # default fallback path (backwards compat), use image tar if root is a .tar file, otherwise image folder
        # FIXME support split here, in parser?
        if os.path.isfile(root) and os.path.splitext(root)[1] == '.tar':
            parser = ParserImageInTar(root, **kwargs)
        else:
            parser = ParserImageFolder(root, **kwargs)
    return parser
