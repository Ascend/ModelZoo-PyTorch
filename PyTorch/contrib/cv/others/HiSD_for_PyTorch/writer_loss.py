# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib.pyplot as plt

def data_read(dir_path):
    x_ = []
    y_ = []
    with open(dir_path, "r") as f:
        raw_data = f.readlines()
    for line in raw_data:
        x,y = line.split(":")
        x_.append(x)
        y_.append(y)
        #data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

if __name__ == "__main__":
    t9rain_loss_path = "/home/test_user06/HiSD-main/train_loss.txt"   # 存储文件路径
    data_x,data_y = data_read(train_loss_path)
    xx = " ".join(data_x)
    yy = " ".join(data_y)
    plt.plot(yy,xx, 'b', label='loss')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.savefig('loss.jpg')