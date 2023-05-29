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
import shutil
from argparse import ArgumentParser


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type", default="npy", help="data type", choices=["npy", "bin"])
    parser.add_argument("--data_root", default="infer_out", help="data root dir")
    parser.add_argument("--driving_dir", default="kpd", help="kp driving save dir")
    parser.add_argument("--source_dir", default="kps", help="kp source save dir")
    opt = parser.parse_args()

    data_root = opt.data_root
    driving_dir = opt.driving_dir
    source_dir = opt.source_dir
    if not data_root[-1] == '/':
        data_root += "/"
    if not driving_dir[-1] == '/':
        driving_dir += "/"
    if not source_dir[-1] == '/':
        source_dir += "/"

    kpd_dir = data_root + driving_dir
    kps_dir = data_root + source_dir

    kp_driving_value_dir = data_root + "kpdv/"
    kp_driving_jac_dir = data_root + "kpdj/"
    kp_source_value_dir = data_root + "kpsv/"
    kp_source_jac_dir = data_root + "kpsj/"
    mkdir(kp_driving_value_dir)
    mkdir(kp_driving_jac_dir)
    mkdir(kp_source_value_dir)
    mkdir(kp_source_jac_dir)
    for i in range(64196):
        if opt.type == "npy":
            kpdv = kpd_dir + str(i) + "_0.npy"
            kpdj = kpd_dir + str(i) + "_1.npy"
            kpsv = kps_dir + str(i) + "_0.npy"
            kpsj = kps_dir + str(i) + "_1.npy"

            kp_driving_value = kp_driving_value_dir + str(i) + ".npy"
            kp_driving_jac = kp_driving_jac_dir + str(i) + ".npy"
            kp_source_value = kp_source_value_dir + str(i) + ".npy"
            kp_source_jac = kp_source_jac_dir + str(i) + ".npy"
        else:
            kpdv = kpd_dir + str(i) + "_0.bin"
            kpdj = kpd_dir + str(i) + "_1.bin"
            kpsv = kps_dir + str(i) + "_0.bin"
            kpsj = kps_dir + str(i) + "_1.bin"

            kp_driving_value = kp_driving_value_dir + str(i) + ".bin"
            kp_driving_jac = kp_driving_jac_dir + str(i) + ".bin"
            kp_source_value = kp_source_value_dir + str(i) + ".bin"
            kp_source_jac = kp_source_jac_dir + str(i) + ".bin"

        shutil.move(kpdv, kp_driving_value)
        shutil.move(kpdj, kp_driving_jac)
        shutil.move(kpsv, kp_source_value)
        shutil.move(kpsj, kp_source_jac)
