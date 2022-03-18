# Copyright 2021 Huawei Technologies Co., Ltd
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
import argparse
import UCAS_AOD_benchmark.data_prepare as prepare
import datasets.evaluate.ucas_aod2gt as ucas_aod2gt


def generate_UCAS_AOD(data_path):
    trainset = os.path.join(data_path, "ImageSets/train.txt")
    valset = os.path.join(data_path, "ImageSets/val.txt")
    testset = os.path.join(data_path, "ImageSets/test.txt")
    img_dir = os.path.join(data_path, "AllImages")
    label_dir = os.path.join(data_path, "Annotations")
    root_dir = data_path

    for dataset in [trainset, valset, testset]:
        with open(dataset, "r") as f:
            names = f.readlines()
            paths = [os.path.join(img_dir, x.strip() + ".png\n") for x in names]
            with open(os.path.join(root_dir, os.path.split(dataset)[1]), "w") as fw:
                fw.write(''.join(paths))


def check_file(opt):
    del_file = ["AllImages", "Annotations", "ImageSets", "Test", "test.txt", "train.txt", "val.txt"]
    for file_name in del_file:
        del_file_path = os.path.join(opt.data_file_path, file_name)
        if os.path.exists(del_file_path):
            raise RuntimeError("file {} already exists!".format(del_file_path))
    imagesets_path = os.path.join(opt.benchmark_path, "ImageSets")
    if os.path.exists(imagesets_path):
        cmd = "cp -r {} {}".format(imagesets_path, opt.data_file_path)
        os.system(cmd)
    else:
        raise RuntimeError("file imagesets_path not exists!".format(imagesets_path))


def prepare_data(opt):
    prepare.creat_tree(opt.data_file_path)
    prepare.generate_test(opt.data_file_path)
    generate_UCAS_AOD(opt.data_file_path)


def prepare_eval_data(opt):
    gt_path = os.path.join(opt.data_file_path, "test.txt")
    ucas_aod2gt.convert_ucas_gt(gt_path, opt.eval_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path",
                        default="./UCAS_AOD",
                        type=str)
    opt = parser.parse_args()
    opt.benchmark_path = "./UCAS_AOD_benchmark"
    opt.eval_path = "./datasets/evaluate/ground-truth"
    check_file(opt)
    prepare_data(opt)
    prepare_eval_data(opt)


if __name__ == '__main__':
    main()
