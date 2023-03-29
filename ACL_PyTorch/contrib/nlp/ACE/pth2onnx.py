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
# ============================================================================
import argparse
import os
import sys

import numpy as np
import torch

sys.path.append('./ACE/')
from flair.config_parser import ConfigParser
from flair.trainers import ReinforcementTrainer
from flair.utils.from_params import Params


def get_model(args):
    config = Params.from_file(args.config)
    config = ConfigParser(config, all=False, zero_shot=False, other_shot=False, predict=False)
    student = config.create_student(nocrf=False)

    corpus = config.corpus

    trainer = ReinforcementTrainer(student, None, corpus, config=config.config, is_test=True)
    base_path = config.get_target_path

    trainer.model = trainer.model.load(base_path / "best-model.pt", device='cpu')
    training_state = torch.load(base_path / 'training_state.pt', map_location=torch.device('cpu'))
    trainer.best_action = training_state['best_action']
    trainer.model.selection = trainer.best_action

    model = trainer.model.to("cpu")
    return model


def run_pth2onnx(args):
    model = get_model(args)

    sentence_tensor = torch.from_numpy(np.random.randn(args.batch_size, 124, 24876).astype("float32"))
    lengths_tensor = torch.tensor(np.random.randint(1, 10, [args.batch_size,]), dtype=torch.int32)
    x = torch.onnx.export(model, (sentence_tensor, lengths_tensor), args.onnx_path,
                          verbose=True,
                          input_names=["sentence_tensor", "lengths_tensor"],
                          output_names=["features"],
                          opset_version=13,
                          keep_initializers_as_inputs=True,
                          )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./')
    parser.add_argument('--onnx_dir', type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    args.onnx_path = "{}/ace_bs{}.onnx".format(args.onnx_dir, args.batch_size)
    if not os.path.exists(args.onnx_dir):
        os.makedirs(args.onnx_dir)
    
    run_pth2onnx(args)
