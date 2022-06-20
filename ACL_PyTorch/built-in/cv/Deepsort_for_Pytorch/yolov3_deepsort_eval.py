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
import os.path as osp
import logging
import argparse
from pathlib import Path

from utils.log import get_logger
from yolov3_deepsort import VideoTracker
from utils.parser import get_config

import motmetrics as mm
mm.lap.default_solver = 'lap'
from utils.evaluation import Evaluator
import acl
def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)

def main(data_root='', seqs=('',), args=""):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    data_type = 'mot'
    result_root = os.path.join(Path(data_root), "mot_results")
    mkdir_if_missing(result_root)

    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    # run tracking
    accs = []
    for seq in seqs:
        logger.info('start seq: {}'.format(seq))
        #result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        result_filename = os.path.join(args.save_path, 'results.txt')
        video_path = data_root+"/"+seq+"/img1/video.mp4"

        with VideoTracker(cfg, args, video_path) as vdo_trk:
            vdo_trk.run()

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_global.xlsx'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=False)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()

if __name__ == '__main__':
    ret = acl.init()
    assert ret == 0
    ret = acl.rt.set_device(0)
    assert ret == 0
    context, ret = acl.rt.create_context(0)
    assert  ret == 0
    args = parse_args()

    seqs_str = '''MOT16-02       
                  MOT16-04
                  MOT16-05
                  MOT16-09
                  MOT16-10
                  MOT16-11
                  MOT16-13
                  '''        
    data_root = 'MOT16/train/'

    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root=data_root,
         seqs=seqs,
         args=args)

    ret = acl.rt.reset_device(0)
    assert ret == 0

    context, ret = acl.rt.get_context()
    assert ret == 0
    ret = acl.rt.destroy_context(context)
    assert ret == 0
    ret = acl.finalize()
    assert ret == 0
