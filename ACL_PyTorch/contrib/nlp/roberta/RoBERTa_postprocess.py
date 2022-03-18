# Copyright 2021 Huawei Technologies Co., Ltd
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
import numpy as np
import argparse
import os


def postProcess(batch_size, res_dir, label_path):
    """calculate acc of the results

    Args:
        batch_size (int): batch size
        res_dir (str): dir of result
        label_path (str): path of label
    """
    labels = np.fromfile(label_path, dtype=np.int64).reshape(-1, batch_size)
    neg_res = 0
    for i in range(len(labels)):
        res_path = os.path.join(
            res_dir, "src_tokens_{}_output_0.bin".format(i))
        res = np.fromfile(res_path, dtype=np.float32).reshape(-1, 2)
        neg_res += sum((np.argmax(res, axis=1)) ^ labels[i])

    print("acc = {:.3f}".format(1 - neg_res/(i*batch_size)))


def gen_res_path(batch_size, device):
    """parse the res path and infer time from log file

    Args:
        batch_size (int): batch size
        device (device): which device to run

    Returns:
        tuple: (path of res, average infer time)
    """
    log_path = "msame_res_bs{}_device{}.log".format(batch_size, device)
    res_path = ""
    with open(log_path, "r") as f:
        for i in f.readlines():
            if "./result" in i:
                res_path = i.strip()
            if "Inference average time : " in i:
                t = eval(i.strip()[len("Inference average time : "): -2])
    return res_path, 1000/(t/batch_size)*4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=0, type=int,
                        help='which device to use')
    parser.add_argument("--data_path", default="./SST-2",
                        type=str, help='dir of data')
    parser.add_argument("--batch_size", default=0, type=int, help='batch size')
    args = parser.parse_args()
    root_path = os.path.join(
        args.data_path, "batch_size_{}".format(args.batch_size))
    label_path = os.path.join(root_path, "roberta_base.label")
    res_path, avg_infer_time = gen_res_path(args.batch_size, args.device)
    postProcess(args.batch_size, res_path, label_path)
    print("fps = {:.3f}".format(avg_infer_time))


if __name__ == "__main__":
    main()
