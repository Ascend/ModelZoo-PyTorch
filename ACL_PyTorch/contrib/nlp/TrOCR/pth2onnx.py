# Copyright 2022 Huawei Technologies Co., Ltd
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
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, 'slim')))
sys.path.append(os.path.abspath(os.path.join(__dir__, 'fairseq')))

from unilm.trocr import task
import argparse
import torch
import fairseq
from fairseq import utils
from fairseq_cli import generate
from PIL import Image
import torchvision.transforms as transforms

def run_pth2onnx(args):
    beam = 10
    device = torch.device('cpu')
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [args.model],
        arg_overrides={"beam": beam, "task": "text_recognition", "data": "", "fp16": False})
    
    model[0].to(device)

    generator = task.build_generator(
        model, cfg.generation, extra_gen_cls_kwargs={'lm_model': None, 'lm_weight': None}
    )

    im = torch.ones([1, 3, 384, 384], dtype=torch.float32).to(device)
    torch.onnx.export(generator, im, 
                      args.onnx_path,
                      input_names=['imgs'],
                      output_names=['cand_bbsz_idx_out', 'eos_mask_out', 'cand_scores_out', 'tokens_out', 'scores_out', 'attn_out'],
                      keep_initializers_as_inputs=True,
                      verbose=False, opset_version=13)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./trocr-small-handwritten.pt')
    parser.add_argument('--onnx_dir', type=str, default='./')
    args = parser.parse_args()
    args.onnx_path = "{}trocr.onnx".format(args.onnx_dir)
    if not os.path.exists(args.onnx_dir):
        os.makedirs(args.onnx_dir)

    run_pth2onnx(args)

    