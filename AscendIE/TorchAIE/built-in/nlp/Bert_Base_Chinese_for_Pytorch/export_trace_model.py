# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import sys
import torch
import numpy as np
from model import build_tokenizer, RefineModel


def generate_random_data(shape, dtype, low=0, high=2):
    if dtype in ["float32", "float16"]:
        return np.random.random(shape).astype(dtype)
    elif dtype in ["int32", "int64"]:
        return np.random.uniform(low, high, shape).astype(dtype)
    else:
        raise NotImplementedError("Not supported dtype: {}".format(dtype))


def export_onnx(model_dir, save_path, seq_len=384, batch_size=1):
    # build tokenizer
    tokenizer = build_tokenizer(tokenizer_name=model_dir)

    # build model
    model_path = os.path.join(model_dir, "bert-base-chinese")
    config_path = os.path.join(model_dir, "config.json")
    model = RefineModel(tokenizer, model_path, config_path)

    # build data
    input_data = (
        torch.Tensor(generate_random_data([batch_size, seq_len], "int64")).to(torch.int64),
        torch.Tensor(generate_random_data([batch_size, seq_len], "int64")).to(torch.int64),
        torch.Tensor(generate_random_data([batch_size, seq_len], "int64")).to(torch.int64)
    )

    ts_model = torch.jit.trace(model, input_data)
    ts_model.save(save_path)


if __name__ == '__main__':
    model_dir = sys.argv[1]
    save_path = sys.argv[2]
    seq_len = int(sys.argv[3])
    export_onnx(model_dir, save_path, seq_len)
