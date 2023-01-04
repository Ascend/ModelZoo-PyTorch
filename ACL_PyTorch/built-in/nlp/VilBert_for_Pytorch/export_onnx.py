# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#

# -*- coding: UTF-8 -*-
import argparse
import torch
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.plugins import import_plugins
import_plugins()


def export_onnx(model, args):
    bsz = args.batch_size
    pdlen = args.pad_len
    box_num = args.box_num

    print("[INFO] Export to onnx.")
    box_features = torch.ones(bsz, box_num, 1024, dtype=torch.float)
    box_coordinates = torch.ones(bsz, box_num, 4, dtype=torch.float)
    box_mask = torch.ones(bsz, box_num, dtype=torch.bool)
    q_token_ids = torch.ones(bsz, pdlen, dtype=torch.int64)
    q_mask = torch.ones(bsz, pdlen, dtype=torch.bool)
    q_type_ids = torch.ones(bsz, pdlen, dtype=torch.int64)
    dummy_input = (
        box_features,
        box_coordinates,
        box_mask,
        q_token_ids,
        q_mask,
        q_type_ids
    )
    dynamic_axes = {
        'box_features': {0: 'batch_size', 1: 'box_num'},
        'box_coordinates': {0: 'batch_size', 1: 'box_num'},
        'box_mask': {0: 'batch_size', 1: 'box_num'},
        'token_ids': {0: 'batch_size', 1: 'seq_len'},
        'mask': {0: 'batch_size', 1: 'seq_len'},
        'type_ids': {0: 'batch_size', 1: 'seq_len'}
    }
    torch.onnx.export(
        model, dummy_input,
        args.save_path,
        dynamic_axes=dynamic_axes,
        input_names=['box_features', 'box_coordinates', 'box_mask',
                     'token_ids', 'mask', 'type_ids'],
        output_names=['logits', 'probs'],
        opset_version=11,
    )
    print('[INFO] Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Evaluate the specified model + dataset""")
    parser.add_argument("--archive_file", type=str, default="models/vilbert-vqa-pretrained.2021-03-15.tar.gz",
                        help="path to an archived trained model")
    parser.add_argument("--weights_file", type=str, help="a path that overrides weights file to use")
    parser.add_argument("--save_path", type=str, help="save path for onnx model") 
    parser.add_argument("--cuda_device", type=int, default=-1, help="id of GPU to use (if any)")
    parser.add_argument("-o", "--overrides", type=str, default="",
                        help="a json(net) structure used to override the experiment configuration, e.g.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="If non-empty, the batch size to use during evaluation.")
    parser.add_argument("--pad_len", type=int, default=32, help="padding sequence to fixed length.")
    parser.add_argument("--box_num", type=int, default=43, help="box num to fixed length.")
    args = parser.parse_args()


    archive = load_archive(
        args.archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )
    model = archive.model
    model.eval()

    export_onnx(model, args)
