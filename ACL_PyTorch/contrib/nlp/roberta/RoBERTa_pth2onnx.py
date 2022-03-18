# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from fairseq import hub_utils
import torch
import argparse
import os

OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX


def pth2onnx(checkpoint_path, checkpoint_file, data_name_or_path, batch_size, pad_length, output_file):
    """convert pth to onnx

    Args:
        checkpoint_path (str): dir of pth
        checkpoint_file (str): name of pth, locate in dir of pth
        data_name_or_path (str): dir of data
        batch_size (int): batch size
        pad_length (int): pad length of sentence
        output_file (str): name of generated onnx file
    """
    model = hub_utils.from_pretrained(
        checkpoint_path,
        checkpoint_file=checkpoint_file,
        data_name_or_path=data_name_or_path,
        bpe="gpt2",
        load_checkpoint_heads=True,
    )["models"][0]
    model.eval()

    org_dummy_input = torch.ones(batch_size, pad_length).long()
    input_names = ["src_tokens"]
    output_names = ["output"]

    torch.onnx.export(model, org_dummy_input, output_file, input_names=input_names,
                      operator_export_type=OPERATOR_EXPORT_TYPE, output_names=output_names, opset_version=11, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path',
                        default="checkpoint/", type=str, help='dir of pth')
    parser.add_argument('--checkpoint_file',
                        default="checkpoint.pt", type=str, help='pth name, locate in  dir of pth')
    parser.add_argument('--onnx_path', default="onnx_file/",
                        type=str, help='dir of onnx')
    parser.add_argument('--data_name_or_path',
                        default="SST-2", type=str, help='dir of data')
    parser.add_argument('--pad_length', default=128, type=int,
                        help='fix the pad length of one sentence')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    args = parser.parse_args()
    ONNX_OUTPUT_PATH = os.path.join(
        args.onnx_path, "roberta_base_batch_{}.onnx".format(args.batch_size))
    pth2onnx(args.checkpoint_path, args.checkpoint_file, args.data_name_or_path,
             args.batch_size, args.pad_length, ONNX_OUTPUT_PATH)
