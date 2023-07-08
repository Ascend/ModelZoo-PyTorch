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


def postProcess(res_dir, label_path, batch_size=1):
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
            res_dir, "src_tokens_{}_0.bin".format(i))
        if os.path.exists(res_path):
            res = np.fromfile(res_path, dtype=np.float32).reshape(-1, 2)
        else:
            res_path = os.path.join(
                res_dir, "src_tokens_{}_0.npy".format(i))
            res = np.load(res_path)
        neg_res += sum((np.argmax(res, axis=1)) ^ labels[i])

    print("acc = {:.3f}".format(1 - neg_res/(i*batch_size)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str,
                        help='infer result dir')
    parser.add_argument("--data_path", default="./SST-2",
                        type=str, help='dir of data')
    args = parser.parse_args()
    label_path = os.path.join(args.data_path, "roberta_base.label")
    postProcess(args.res_path, label_path)


if __name__ == "__main__":
    main()
