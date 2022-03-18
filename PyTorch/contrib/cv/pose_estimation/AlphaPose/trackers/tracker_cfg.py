# Copyright 2020 Huawei Technologies Co., Ltd
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
from easydict import EasyDict as edict
cfg = edict()
cfg.nid = 1000
cfg.arch = "osnet_ain" # "osnet" or "res50-fc512"
cfg.loadmodel = "trackers/weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
cfg.frame_rate =  30
cfg.track_buffer = 240 
cfg.conf_thres = 0.5
cfg.nms_thres = 0.4
cfg.iou_thres = 0.5
