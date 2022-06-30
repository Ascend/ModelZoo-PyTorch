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
import socket
import subprocess

from train import main as single_process_main
from fairseq import distributed_utils, options


def main(args):
    if args.distributed_init_method is None and args.distributed_port > 0:
        # We can determine the init method automatically for Slurm.
        node_list = os.environ.get('SLURM_JOB_NODELIST')
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
                args.distributed_init_method = 'tcp://{host}:{port}'.format(
                    host=hostnames.split()[0].decode('utf-8'),
                    port=args.distributed_port)
                args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
                args.device_id = int(os.environ.get('SLURM_LOCALID'))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError as e:  # Slurm is not installed
                pass
    if args.distributed_init_method is None and args.distributed_port is None:
        raise ValueError('--distributed-init-method or --distributed-port '
                         'must be specified for distributed training')

    args.distributed_rank = distributed_utils.distributed_init(args)
    print('| initialized host {} as rank {}'.format(socket.gethostname(), args.distributed_rank))
    single_process_main(args)


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
