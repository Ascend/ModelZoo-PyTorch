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
import pickle as pk

const_img_shape = (608, 608)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate dataset info file')
    parser.add_argument("--bin_file_path", default="val2017_bin")
    parser.add_argument("--meta_file_path", default="val2017_bin_meta")
    parser.add_argument("--bin_info_file_name", default="yolof.info")
    parser.add_argument("--meta_info_file_name", default="yolof_meta.info")
    args = parser.parse_args()

    with open(args.bin_info_file_name, "w") as fp1, open(args.meta_info_file_name, "w") as fp2:
        file_list = os.listdir(args.bin_file_path)
        idx = 0
        for file in file_list:
            fp1.write("{} {}/{} {} {}\n".format(idx, args.bin_file_path, file, *const_img_shape))
            file_name = file.split(".")[0]
            with open("{}/{}.pk".format(args.meta_file_path, file_name), "rb") as fp_t:
                meta = pk.load(fp_t)
                fp2.write("{} {} {} {}\n".format(idx, file_name, *meta["ori_shape"]))
            idx += 1
    print("Get info done!")
