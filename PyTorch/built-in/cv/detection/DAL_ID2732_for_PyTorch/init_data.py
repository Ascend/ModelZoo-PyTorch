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
import argparse


def init_file(opt):
    del_file = ["AllImages", "Annotations", "ImageSets", "Test", "test.txt", "train.txt", "val.txt"]
    for file_name in del_file:
        del_file_path = os.path.join(opt.data_file_path, file_name)
        if os.path.exists(del_file_path):
            cmd = "rm -rf {}".format(del_file_path)
            os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path",
                        default="./UCAS_AOD",
                        type=str)
    opt = parser.parse_args()
    init_file(opt)


if __name__ == '__main__':
    main()
