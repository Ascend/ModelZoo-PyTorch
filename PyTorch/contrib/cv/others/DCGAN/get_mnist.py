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
import torchvision.datasets as datasets
import argparse

parser = argparse.ArgumentParser(description="MNIST dataset")
parser.add_argument('--data_path', metavar='DIR', type=str, default="./data",
                    help='path to dataset')

if __name__ == "__main__":
    args = parser.parse_args()
    print("MNIST target folder : ", args.data_path)
    print("start download...")
    train_dataset = datasets.MNIST(
        args.data_path,
        train=True,
        download=True)
    print("download done...")
