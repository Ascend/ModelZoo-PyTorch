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


from __future__ import print_function

import argparse
import os

import torch
import onnx, onnxruntime
import yaml
import numpy as np

from wenet.transformer.asr_model import init_asr_model
from wenet.transformer.decoder import TransformerDecoder, BiTransformerDecoder
from wenet.utils.checkpoint import load_checkpoint


def to_numpy(xx):
    return xx.detach().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    # parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument('--output_onnx_file', required=True, help='output onnx file')
    args = parser.parse_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_asr_model(configs)
    print(model)

    load_checkpoint(model, args.checkpoint)
    # Export jit torch script model

    model.eval()

    #export the none flash model
    encoder = model.encoder
    xs = torch.randn(1, 131, 80, requires_grad=False)
    xs_lens = torch.tensor([131], dtype=torch.int32)
    onnx_encoder_path = os.path.join(args.output_onnx_file, 'no_flash_encoder.onnx')
    torch.onnx.export(encoder,
                    (xs, xs_lens),
                    onnx_encoder_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['xs_input', 'xs_input_lens'],
                    output_names=['xs_output', 'masks_output'],
                    dynamic_axes={'xs_input': [1], 'xs_input_lens': [0],
                                    'xs_output': [1], 'masks_output': [2]},
                    verbose=True
                    )
    onnx_model = onnx.load(onnx_encoder_path)
    onnx.checker.check_model(onnx_model)
    print("encoder onnx_model check pass!")

    ort_session = onnxruntime.InferenceSession(onnx_encoder_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(xs),
                ort_session.get_inputs()[1].name: to_numpy(xs_lens),
                }
    ort_outs = ort_session.run(None, ort_inputs)
    y1, y2 = encoder(xs, xs_lens)
    # np.testing.assert_allclose(to_numpy(y1), ort_outs[0], rtol=1e-05, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(y2), ort_outs[1], rtol=1e-05, atol=1e-05)
    print("Exported no flash encoder model has been tested with ONNXRuntime, and the result looks good!")

    #export the flash encoder
    encoder = model.encoder
    encoder.forward = encoder.forward_chunk

    batch_size = 1
    audio_len = 131
    x = torch.randn(batch_size, audio_len, 80, requires_grad=False)
    offset = torch.tensor(1)
    decoding_chunk_size = 16
    num_decoding_left_chunks = -1
    required_cache_size = decoding_chunk_size * num_decoding_left_chunks
    required_cache_size = torch.tensor(required_cache_size)
    subsampling_cache = torch.randn(batch_size, 1, 256, requires_grad=False)
    elayers_cache = torch.randn(12, batch_size, 1, 256, requires_grad=False)
    conformer_cnn_cache = torch.randn(12, batch_size, 256, 7, requires_grad=False)


    encoder.set_onnx_mode(False)
    y, subsampling_cache_output, elayers_cache_output, conformer_cnn_cache_output = encoder(x, torch.tensor(0), \
                                                                required_cache_size, None, None, conformer_cnn_cache)

    encoder.set_onnx_mode(True)
    onnx_encoder_path = os.path.join(args.output_onnx_file, 'encoder.onnx')
    torch.onnx.export(encoder,
                    (x, offset, required_cache_size, subsampling_cache, elayers_cache, conformer_cnn_cache),
                    onnx_encoder_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input', 'offset', 'required_cache_size', 'subsampling_cache', 'elayers_cache', \
                                 'conformer_cnn_cache'],
                    output_names=['output', 'subsampling_cache_output', 'elayers_cache_output', \
                                  'conformer_cnn_cache_output'],
                    dynamic_axes={'input': [1], 'subsampling_cache':[1], 'elayers_cache':[2],
                                    'output': [1]},
                    verbose=True
                    )

    onnx_model = onnx.load(onnx_encoder_path)
    onnx.checker.check_model(onnx_model)
    print("encoder onnx_model check pass!")

    ort_session = onnxruntime.InferenceSession(onnx_encoder_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x),
                ort_session.get_inputs()[1].name: to_numpy(offset),
                ort_session.get_inputs()[2].name: to_numpy(subsampling_cache),
                ort_session.get_inputs()[3].name: to_numpy(elayers_cache),
                ort_session.get_inputs()[4].name: to_numpy(conformer_cnn_cache),
                }
    ort_outs = ort_session.run(None, ort_inputs)
    print("Exported encoder model has been tested with ONNXRuntime, and the result looks good!")

    #export decoder onnx

    decoder = model.decoder
    decoder.set_onnx_mode(True)
    onnx_decoder_path = os.path.join(args.output_onnx_file, 'decoder.onnx')
    memory = torch.randn(10, 131, 256)
    memory_mask = torch.ones(10, 1, 131).bool()
    ys_in_pad = torch.randint(0, 4232, (10, 50)).long()
    ys_in_lens = torch.tensor([13, 13, 13, 13, 13, 13, 13, 13, 50, 13], dtype=torch.int32)
    r_ys_in_pad = torch.randint(0, 4232, (10, 50)).long()

    if isinstance(decoder, TransformerDecoder):
        torch.onnx.export(decoder,
                        (memory, memory_mask, ys_in_pad, ys_in_lens),
                        onnx_decoder_path,
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=['memory', 'memory_mask', 'ys_in_pad', 'ys_in_lens'],
                        output_names=['l_x', 'r_x'],
                        dynamic_axes={'memory': [1], 'memory_mask':[2], 'ys_in_pad':[1],
                                        'ys_in_lens': [0]},
                        verbose=True
                        )
    elif isinstance(decoder, BiTransformerDecoder):
        print("BI mode")
        torch.onnx.export(decoder,
                        (memory, memory_mask, ys_in_pad, ys_in_lens, r_ys_in_pad),
                        onnx_decoder_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['memory', 'memory_mask', 'ys_in_pad', 'ys_in_lens', 'r_ys_in_pad'],
                        output_names=['l_x', 'r_x', 'olens'],
                        dynamic_axes={'memory': [1], 'memory_mask':[2], 'ys_in_pad':[1],
                                        'ys_in_lens': [0], 'r_ys_in_pad':[1]},
                        verbose=True
                        )

