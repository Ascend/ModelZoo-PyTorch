# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import os.path as osp
import logging
import motmetrics as mm
import json
import numpy as np
import glob
import sys

sys.path.insert(0, "./FairMOT/src")
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.log import logger
import lib.datasets.dataset.jde as datasets
from lib.tracking_utils.evaluation import Evaluator
from lib.opts import opts
from track import write_results
from lib.tracking_utils.timer import Timer
from lib.tracker.multitracker import JDETracker

def process(opt,
            seqs, 
            data_root,
            input_root, 
            exp_name='MOT17_test_public_dla34',
            show_image=False,
            save_images=False,
            save_videos=False):


    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    with open(os.path.join(input_root, 'sumary.json'), 'r') as f:
        json_data = json.load(f)
        json_info = json_data['filesinfo']
        file_dict = {}
        for k, x in json_info.items():
            file_dict[x['infiles'][0]] = x['outfiles']
    
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        # all_data = sorted(glob.glob(os.path.join(input_root, '*.npy')))
        dataloader = []
        for k, v in file_dict.items():
            if seq in k:
                dataloader.extend(v)
        logger.info("{} seq {} images".format(seq, len(dataloader)))
        # dataloader = list(filter(lambda x: os.path.split(x)[1].split("_")[0] == seq, all_data)) 
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename, seq,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)


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
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))

def eval_seq(opt, dataloader, data_type, result_filename, seq, save_dir=None, 
                show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    #for path, img, img0 in dataloader:
    for i in range(0, len(dataloader), 4):
        # logger.info("{}, {}, {}, {}".format(dataloader[i + 3],dataloader[i + 2],dataloader[i + 1], dataloader[i]))

        hm_eval = torch.from_numpy(np.fromfile(dataloader[i], dtype='float32').reshape(1, 1, 152, 272))
        wh_eval = torch.from_numpy(np.fromfile(dataloader[i + 1], dtype='float32').reshape(1, 4, 152, 272))
        id_eval = torch.from_numpy(np.fromfile(dataloader[i + 2], dtype='float32').reshape(1, 128, 152, 272))
        reg_eval = torch.from_numpy(np.fromfile(dataloader[i+ 3], dtype='float32').reshape(1, 2, 152, 272))
    
        timer.tic()
        if seq == "MOT17-05-SDP":
            img0 = np.zeros([480, 640])
        else:
            img0 = np.zeros([1080, 1920])
        online_targets = tracker.update(hm_eval, wh_eval, id_eval, reg_eval, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        frame_id += 1

    # save results
    write_results(result_filename, results, data_type)

    return frame_id, timer.average_time, timer.calls



if __name__ == "__main__":
    opt = opts().init()
    seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
    
    input_root = opt.input_root
    data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    seqs = [seq.strip() for seq in seqs_str.split()]
    process(opt,
            seqs,
            data_root,
            input_root,
            exp_name='MOT17_test_public_dla34',
            show_image=False,
            save_images=False,
            save_videos=False)
