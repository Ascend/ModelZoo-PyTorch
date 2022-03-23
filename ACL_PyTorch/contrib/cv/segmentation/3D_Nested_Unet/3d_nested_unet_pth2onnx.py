# Copyright 2020 Huawei Technologies Co., Ltd
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

# 3d_nested_unet_pth2onnx.py
import sys
import os
import time
import pdb
import argparse
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.inference.predict2 import pth2onnx


def main():
    # pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--file_path', help='output onnx file path', required=True)
    args = parser.parse_args()
    fp = args.file_path
    model = '3d_fullres'
    task_name = 'Task003_Liver'
    trainer = 'nnUNetPlusPlusTrainerV2'
    plans_identifier = 'nnUNetPlansv2.1'
    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" + plans_identifier)
    model = model_folder_name
    folds = None  # 如果文件存放路径正确，会自动识别到教程中的fold 0
    mixed_precision = True
    checkpoint_name = 'model_final_checkpoint'
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision, checkpoint_name=checkpoint_name)
    pre_mode = -1
    if int(pre_mode) == -1:
        p = params[0]
        trainer.load_checkpoint_ram(p, False)  # nnUnetPlusPlusTrainerV2，实际函数在network_trainer里
        print('pth2onnx start')
        pth2onnx(trainer.network, fp)
        print('pth2onnx end')
        print('onnx模型已经输出至：', fp)


if __name__ == "__main__":
    main()
    print('main end')


