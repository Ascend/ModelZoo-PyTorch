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
import sys,datetime

# 解析输入参数data_url
parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/home/ma-user/modelarts/inputs/data_url_0")
parser.add_argument("--train_url", type=str, default="/home/ma-user/modelarts/outputs/train_url_0/")
parser.add_argument("--model_sex", type=str, default="males")
parser.add_argument("--init_env", type=str, default="True")
parser.add_argument("--debug", type=str, default="False")

config = parser.parse_args()

def run(cmd , log_loc=None, ):
    """显示终端输入"""
    if log_loc:
        cmd = cmd + " >> "+ log_loc
    print(f"> {cmd}")
    result = os.system(cmd)
    return result == 0

def init_dir(dirpaths):
    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

def printC(arg,value,num=30):
    arg_num = len(arg)
    if arg_num < num:
        arg = arg + " "*(num-arg_num)
    print("{}: {}".format(arg,value))

if __name__ == '__main__':

    DEBUG = eval(config.debug)
    #获取文件位置
    cur_dir = os.path.dirname(__file__)
    script_dir = os.path.join(cur_dir, "run_scripts")
    train_script_dir = os.path.join(script_dir, "train")
    train_sh_loc = os.path.join(train_script_dir, "train.sh")
    init_env_sh_loc = os.path.join(train_script_dir, "preenv.sh")

    ROOT_DIR = os.path.dirname(cur_dir)

    DATA_ROOT = config.data_url
    TRAIN_ROOT = config.train_url

    LOG_DIR = os.path.join(TRAIN_ROOT, "log")
    CHECKPOINTS_DIR = os.path.join(TRAIN_ROOT, "checkoutpoints")
    DATASET_ROOT = os.path.join(DATA_ROOT, config.model_sex)

    init_dir([LOG_DIR,CHECKPOINTS_DIR])

    TRAIN_SCRPIT_LOC=os.path.join(train_script_dir, "train.sh")

    printC("Root Dir", ROOT_DIR)
    printC("Cur dir", cur_dir)
    printC("Script dir", script_dir)
    printC("Train shell script", train_sh_loc)
    printC("Initial env shell script", init_env_sh_loc)
    printC("Data Root", DATA_ROOT)
    printC("Train Root", TRAIN_ROOT)

    print("-"*50)

    printC("Log Dir", LOG_DIR)
    printC("Checkpoints Dir", CHECKPOINTS_DIR)
    printC("Dataset Dir", DATASET_ROOT)


    # run("ls -ahl %s" %cur_dir)
    run("pip install torchsummary")
    print("-"*50)

    if eval(config.init_env):
        #运行初试化环境脚本 run_scripts/train/preenv.sh
        INIT_ENV_CMD = "bash {} {} {}".format(
            init_env_sh_loc,
            DATA_ROOT,
            TRAIN_ROOT,
        )
        printC("Initial Env Cmd", INIT_ENV_CMD)
        INIT_LOG_NAME = os.path.join(LOG_DIR, datetime.datetime.now().strftime("initenv_%H-%M-%S_%Y-%m-%d.log"))
        printC("Initial Env Log", INIT_LOG_NAME)
        if not run(INIT_ENV_CMD, INIT_LOG_NAME):
            printC("Initial Env Status","Failed")
            exit(1)


    TRAIN_LOG_NAME = os.path.join(LOG_DIR, datetime.datetime.now().strftime("train_%H-%M-%S_%Y-%m-%d.log"))
    printC("Train Log", TRAIN_LOG_NAME)
    TRAIN_CMD = "bash {} {} {} {}".format(
        TRAIN_SCRPIT_LOC,
        DATA_ROOT,
        TRAIN_ROOT,
        config.model_sex,
    )
    printC("Train Cmd", TRAIN_CMD)
    if not run(TRAIN_CMD, TRAIN_LOG_NAME):
        print("Train Status", "Failed")
        exit(1)
    print("-"*50)

