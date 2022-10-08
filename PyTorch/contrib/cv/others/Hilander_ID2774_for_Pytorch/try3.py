# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
import os
import argparse
import sys

# 解析输入参数data_url
parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/home/ma-user/modelarts/inputs/data_url_0/")
parser.add_argument("--train_url", type=str, default="/home/ma-user/modelarts/outputs/train_url_0/")
config = parser.parse_args()


def run(cmd):
    print(f">{cmd}")
    os.system(cmd)




if __name__ == '__main__':
    
    current_path = os.path.dirname(__file__)

    sys.path.append(current_path)
    path1 = os.path.join(current_path, "clustering-benchmark-master")

    os.chdir(path1)
    run("python setup.py install")
    os.chdir(current_path)
    current_path = os.path.dirname(__file__)
    path2 = os.path.join(current_path, "dgl1", "python")
    os.chdir(path2)
    os.system("python setup.py install")
    os.system("python -c 'import dgl; print(dgl.__version__)'")
    os.chdir(current_path)

    # 初始化终端(不要修改) 开始
    code_dir = sys.path[0]
    print("[CANN-Modelzoo] code_dir path is [%s]" % code_dir)
    os.chdir(code_dir)  # cd code_dir
    run("ls -ahl %s" % code_dir)
    # 初始化终端 结束
    # 查看数据集是否配置正确(不要修改) 开始
    print("[CANN-Modelzoo] dataset: %s" % config.data_url)
    # run("ls -ahl %s" % config.data_url)
    run("ls -ahl %s" % config.train_url)

    run("npu-smi info")
    # 查看数据集是否配置正确 结束
    # 加入、添加下面代码到你的脚本(xx.py)中以支持传参(必须修改) 开始

    # 修改你的脚本以支持传递数据集等位置 结束
    print(f"{'#' * 30} your log start {'#' * 30}")
    # 执行训练(按需修改) 开始
    training_script = "cd %s && python train_subg.py --data_path=%s  --model_filename  checkpoint/inat_resampled_1_in_6_per_class.ckpt --knn_k 10,5,3 --levels 2,3,4 --hidden 512 --epochs 250 --lr 0.01 --batch_size 4096 --num_conv 1 --gat --balance" % (
        code_dir, config.data_url + 'inat2018_train_dedup_inter_intra_1_in_6_per_class.pkl')
    print("[CANN-Modelzoo] start run training script:")
    run(training_script)
    # 执行训练 结束
    print(f"{'#' * 30} your log end {'#' * 30}")
    # 将当前执行目录所有文件拷贝到obs进行备份(不要修改) 开始
    print("[CANN-Modelzoo] finish run train shell")
    run("cp -r %s %s " % (code_dir, config.train_url))
    # 将当前执行目录所有文件拷贝到obs进行备份 结束
    # 查看obs是否接收(不要修改) 开始
    print("[CANN-Modelzoo] after train - list my obs backup files:")
    run("ls -al %s" % config.train_url)
    # 查看obs是否接收 结束
    print("Done")
