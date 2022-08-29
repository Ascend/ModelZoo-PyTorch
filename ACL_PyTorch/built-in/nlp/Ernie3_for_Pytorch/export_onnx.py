# Copyright 2022 Huawei Technologies Co., Ltd
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
from lib2to3.pgen2.tokenize import tokenize
from paddlenlp.transformers import *
import torch
import argparse


def export(model_name, model_type, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_type == "AutoModelForSequenceClassification":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if model_type == "AutoModelForTokenClassification":
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    if model_type == "AutoModelForQuestionAnswering":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    if model_type == "AutoModel":
        model = AutoModel.from_pretrained(model_name)
    model.eval()
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[-1, -1],
                                    dtype="int64"),
            paddle.static.InputSpec(shape=[-1, -1],
                                    dtype="int64"),
        ]
    )

    save_path = os.path.join(save_path, "inference")
    paddle.jit.save(model, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("yolox tiny generate quant data")
    parser.add_argument('--model_name', type=str,
                        default="ernie-3.0-base-zh", help='base medium or others')
    parser.add_argument('--model_type', type=str, default="AutoModelForSequenceClassification",
                        choices=["AutoModel", "AutoModelForSequenceClassification", "AutoModelForTokenClassification",
                                 "AutoModelForQuestionAnswering"], help='model type')
    parser.add_argument('--save_path', type=str,
                        default="./ernie", help='static jit path')
    args = parser.parse_args()
    export(args.model_name, args.model_type, args.save_path)
