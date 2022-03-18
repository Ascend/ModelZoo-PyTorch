# Copyright 2020 Huawei Technologies Co., Ltd
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
import os
import torch
import torch.onnx
import argparse
from transformer.modeling import TinyBertForSequenceClassification


def make_input(args):
    """make the input data to create a model"""
    eval_batch_size = args.eval_batch_size
    max_seq_length = args.max_seq_length
    org_input_ids = torch.ones(eval_batch_size, max_seq_length).long()
    org_token_type_ids = torch.ones(eval_batch_size, max_seq_length).long()
    org_input_mask = torch.ones(eval_batch_size, max_seq_length).long()
    return (org_input_ids, org_token_type_ids, org_input_mask)


def convert(args):
    """convert the files into data"""
    model = TinyBertForSequenceClassification.from_pretrained(args.input_model, num_labels = 2)
    model.eval()
    org_input = make_input(args)
    input_names = ['input_ids', 'segment_ids', 'input_mask']
    output_names = ['output']
    OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
    torch.onnx.export(model, org_input, args.output_file, export_params = True,
        input_names=input_names, output_names=output_names,
        operator_export_type=OPERATOR_EXPORT_TYPE,
        opset_version=11, verbose=True)


def main():
    """change the pth files into onnx"""
    #set the args list
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The model(e.g. SST-2 distilled model)dir.")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file of onnx. File name or dir is available.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    convert(args)
    #add_cast(args)

if __name__ == "__main__":
    main()
