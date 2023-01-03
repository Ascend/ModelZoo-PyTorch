# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

import json
import logging
import os
import subprocess
from argparse import ArgumentParser


logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split("=")[0])

    return parser.parse_args()


def main():
    args = parse_args()
    port = 8888
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    num_nodes = len(hosts)
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    os.environ["NCCL_DEBUG"] = "INFO"

    if num_nodes > 1:
        cmd = f"""python -m torch.distributed.launch \
                --nnodes={num_nodes}  \
                --node_rank={rank}  \
                --nproc_per_node={num_gpus}  \
                --master_addr={hosts[0]}  \
                --master_port={port} \
                ./run_glue.py \
                {"".join([f" --{parameter} {value}" for parameter,value in args.__dict__.items()])}"""
    else:
        cmd = f"""python -m torch.distributed.launch \
            --nproc_per_node={num_gpus}  \
            ./run_glue.py \
            {"".join([f" --{parameter} {value}" for parameter,value in args.__dict__.items()])}"""
    try:
        subprocess.run(cmd, shell=True)
    except Exception as e:
        logger.info(e)


if __name__ == "__main__":
    main()
