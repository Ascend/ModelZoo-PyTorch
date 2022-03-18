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


import argparse
import os
import re
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

import apex
from apex import amp

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("--use_npu", dest="use_npu", default=True, action="store_true", help="ues-npu for eval")
    parser.add_argument("--eval_last_15epoch", dest="eval_last_15epoch", default=True, action="store_true", help="eval_last_15epoch for eval")
    parser.add_argument("--cf", default=None, type=str, help="ckpt files for eval")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1
    #is_distributed = True

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)

    torch.npu.set_device(0)
    model.npu()
    model = amp.initialize(model, opt_level='O1')
    model.eval()

    best_ap = -1
    best_epoch = 0
    best_epoch_path = ""
    if not args.speed and not args.trt:
        if args.cf is None:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint from {}".format(ckpt_file))
            loc = "npu:{}".format(rank)
            ckpt = torch.load(ckpt_file, map_location=loc)
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

            if is_distributed:
                model = DDP(model, device_ids=[rank])

            if args.fuse:
                logger.info("\tFusing model...")
                model = fuse_model(model)
            if args.trt:
                assert (
                    not args.fuse and not is_distributed and args.batch_size == 1
                ), "TensorRT model is not support model fusing and distributed inferencing!"
                trt_file = os.path.join(file_name, "model_trt.pth")
                assert os.path.exists(
                    trt_file
                ), "TensorRT model is not found!\n Run tools/trt.py first!"
                model.head.decode_in_inference = False
                decoder = model.head.decode_outputs
            else:
                trt_file = None
                decoder = None

            # start evaluate
            *_, summary = evaluator.evaluate(
                model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
            )
            logger.info("\n" + summary)    
        else:
            files = os.listdir(args.cf)
            files.sort()
            num_ckpts = 0
            for cur_epoch, cfile in enumerate(files):
                # Only evaluate the last 15 ckpts.
                if not "latest_" in str(cfile):
                    continue
                if args.eval_last_15epoch and int(re.findall(r"\d+", cfile)[0]) <= 285:
                    continue
                num_ckpts += 1
                ckpt_file = cfile
                logger.info("loading checkpoint from {}".format(ckpt_file))
                loc = "npu:{}".format(rank)
                ckpt = torch.load(args.cf+"/"+ckpt_file, map_location='cpu')
                model.load_state_dict(ckpt["model"])
                model.to(loc)
                logger.info("loaded checkpoint done.")

                if is_distributed:
                    model = DDP(model, device_ids=[rank])

                if args.fuse:
                    logger.info("\tFusing model...")
                    model = fuse_model(model)

                if args.trt:
                    assert (
                        not args.fuse and not is_distributed and args.batch_size == 1
                    ), "TensorRT model is not support model fusing and distributed inferencing!"
                    trt_file = os.path.join(file_name, "model_trt.pth")
                    assert os.path.exists(
                        trt_file
                    ), "TensorRT model is not found!\n Run tools/trt.py first!"
                    model.head.decode_in_inference = False
                    decoder = model.head.decode_outputs
                else:
                    trt_file = None
                    decoder = None

                logger.info("-"*30)
                logger.info("Result of epoch {}".format(cur_epoch+1))
                # start evaluate
                #*_, summary = evaluator.evaluate(model, is_distributed, args.fp16, trt_file, decoder, exp.test_size )
                ap50_95, ap50, summary = exp.eval(model, evaluator, is_distributed)

                logger.info("\n" + summary)
                logger.info("-"*30)
                if best_ap < ap50_95:
                    best_ap = ap50_95
                    best_epoch = cur_epoch
                    best_epoch_path = args.cf + "/" + cfile
            if num_ckpts == 0:
                logger.info("Not ckpts after 285epochs found, set eval_last_15epoch as False for other ckpts.")
            if best_ap != -1:
                target_path = args.cf + "/" + "best_ckpt.pth"
                os.system("cp " + best_epoch_path +" "+ target_path)
                ckpt_file = best_epoch_path
                logger.info("Loading checkpoint from best_ckpt.pth.")
                loc = "npu:{}".format(rank)
                ckpt = torch.load(ckpt_file, map_location=loc)
                model.load_state_dict(ckpt["model"])
                logger.info("Loading checkpoint done.")
                if is_distributed:
                    model = DDP(model, device_ids=[rank])
                if args.fuse:  
                    logger.info("\tFusing model...")
                    model = fuse_model(model)
            
                if args.trt:
                    assert (
                        not args.fuse and not is_distributed and args.batch_size == 1
                    ), "TensorRT model is not support model fusing and distributed inferencing!"
                    trt_file = os.path.join(file_name, "model_trt.pth")
                    assert os.path.exists(
                        trt_file
                    ), "TensorRT model is not found!\n Run tools/trt.py first!"
                    model.head.decode_in_inference = False
                    decoder = model.head.decode_outputs
                else:
                    trt_file = None
                    decoder = None

                logger.info("*"*30)
                logger.info("The best result is epoch {}".format(best_epoch+1))
                # start evaluate
                *_, summary = evaluator.evaluate(
                    model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
                )
                logger.info("\n" + summary)
                logger.info("*"*30)
            

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.npu.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.npu.device_count()
    print(num_gpu)
    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
