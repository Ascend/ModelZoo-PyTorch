# Copyright 2022 Huawei Technologies Co., Ltd
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


import os
import sys
import re
import os.path as osp

import numpy as np
import torch

sys.path.append(r"slowfast")
from slowfast.utils import distributed as du
from slowfast.utils import logging as logging
from slowfast.utils.meters import TestMeter
from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job

logger = logging.get_logger(__name__)


def perform_x3d_postprocess(test_meter, om_preds_path, cfg):
    bin_list = [file for file in os.listdir(om_preds_path)
                if not file.endswith('json')]
    bin_list.sort(key = lambda i: int(re.match(r'(\d+)', i).group()))
    cfg.TEST.BATCH_SIZE = 1

    for cur_iter, bin in enumerate(bin_list):
        test_meter.data_toc() 
        preds = np.fromfile(osp.join(om_preds_path, bin), dtype=np.float32)
        preds = torch.from_numpy(preds)
        preds = preds.view(1, 400)
        labels = torch.tensor([int(bin.split(".")[0].split("_")[1])])
        video_index = torch.tensor([cur_iter])

        test_meter.iter_toc()
        test_meter.update_stats(preds, labels, video_index)
        test_meter.iter_tic()
    
    test_meter.finalize_metrics()
    return test_meter


def x3d_postprocess(cfg):
    logging.setup_logging(cfg.OUTPUT_DIR)

    om_preds_path = cfg.X3D_POSTPROCESS.OM_OUTPUT_PATH
    bin_list = [file for file in os.listdir(om_preds_path)
                if not file.endswith('json')]
    num_videos = len(bin_list)
    test_meter = TestMeter(
        num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        num_videos,
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    perform_x3d_postprocess(test_meter, om_preds_path, cfg)


if __name__== '__main__':
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    launch_job(cfg=cfg, init_method=args.init_method, func=x3d_postprocess)
