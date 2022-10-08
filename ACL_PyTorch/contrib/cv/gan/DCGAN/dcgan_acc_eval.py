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

"""
generate accuracy value by comparing 'om result' with 'pth result'.
"""

import argparse
import os

import numpy as np


def get_array(pth_bin_path, om_bin_path):
    pth_array = np.fromfile(pth_bin_path, dtype=np.float32)
    om_array = np.fromfile(om_bin_path, dtype=np.float32)

    assert pth_array.shape == om_array.shape, (
            "pth_array(%s) != om_array(%s)" % (str(pth_array.shape), str(om_array.shape)))

    return pth_array, om_array


def cosine_similarity(a, b):
    """
    calculate consine value of vector 'a' and vector'b'
    :param a:vector(ndarray)
    :param b: vector(ndarray)
    :return: cosine value
    """
    assert a.ndim == 1 and b.ndim == 1, "array's ndim is worry"

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    assert a_norm != 0, "the L2 norm of a is zero"
    assert b_norm != 0, "the L2 norm of b is zero"

    similarity = np.dot(a, b.T) / (a_norm * b_norm)
    return similarity


def get_mean_and_cos(a, b):
    mean = np.mean(np.fabs(a - b))
    cos = cosine_similarity(a, b)

    return mean, cos


def main(pth_result_path, om_result_path, log_save_name):
    img_name_list = os.listdir(pth_result_path)
    img_name_list.sort()
    samples_len = len(img_name_list)

    all_mean = 0
    all_cos = 0
    for name in img_name_list:
        pth_bin_path = os.path.join(pth_result_path, name)
        om_bin_path = os.path.join(om_result_path, name)

        pth_array, om_array = get_array(pth_bin_path, om_bin_path)
        mean_tmp, cos_tmp = get_mean_and_cos(pth_array, om_array)

        print("{:s} : mean ==> {:.4f} , cos ==> {:.4f}".format(name, mean_tmp, cos_tmp))

        all_mean += mean_tmp
        all_cos += cos_tmp
    final_mean = all_mean / samples_len
    final_cos = all_cos / samples_len

    res_str = "mean : {:.4f}, cosine : {:.4f}, acc : {:.2f}%".format(
        final_mean, final_cos, 100 * (final_cos + 1) / 2)
    print(res_str)
    with open(log_save_name, "w") as f:
        f.write("all_mean : {:.4f} all_cos : {:.4f}\n".format(all_mean, all_cos))
        f.write(res_str + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_result_path", default="./pth_result/", type=str)
    parser.add_argument("--om_result_path", default="./result/dumpOutput_device0/", type=str)
    parser.add_argument("--log_save_name", default="dcgan_acc_eval.log", type=str)
    args = parser.parse_args()

    main(args.pth_result_path, args.om_result_path, args.log_save_name)
