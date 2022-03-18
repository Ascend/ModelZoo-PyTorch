#
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
#
"""Runs inference on clips much longer than 1s, by running a sliding window and aggregating predictions."""


from argparse import ArgumentParser
from config_parser import get_config
import torch
import numpy as np
import librosa
from utils.misc import get_model

from tqdm import tqdm
import os
import glob
import json
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def process_window(x, sr, audio_settings):
    x = librosa.util.fix_length(x, sr)
    x = librosa.feature.melspectrogram(y=x, **audio_settings)        
    x = librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=audio_settings["n_mels"])
    return x

@torch.no_grad()
def get_clip_pred(net, audio_path, win_len, stride, thresh, config, batch_size, device, mode, label_map) -> list:
    """Performs clip-level inference."""

    net.eval()
    preds_list = []

    audio_settings = config["hparams"]["audio"]
    sr = audio_settings["sr"]
    win_len, stride = int(win_len * sr), int(stride * sr)
    x = librosa.load(audio_path, sr)[0]

    windows, result = [], []

    slice_positions = np.arange(0, len(x) - win_len + 1, stride)

    for b, i in enumerate(slice_positions):
        windows.append(
            process_window(x[i: i + win_len], sr, audio_settings)
        )

        if (not (b + 1) % batch_size) or (b + 1) == len(slice_positions):
            windows = torch.from_numpy(np.stack(windows)).float().unsqueeze(1)
            windows  = windows.to(f'npu:{NPU_CALCULATE_DEVICE}')
            out = net(windows)
            conf, preds = out.softmax(1).max(1)
            conf, preds = conf.cpu().numpy().reshape(-1, 1), preds.cpu().numpy().reshape(-1, 1)

            starts = slice_positions[b - preds.shape[0] + 1: b + 1, None]
            ends = starts + win_len

            res = np.hstack([preds, conf, starts, ends])
            res = res[res[:, 1] > thresh].tolist()
            if len(res):
                result.extend(res)
            windows = []

    #######################
    # pred aggregation
    #######################
    pred = []
    if len(result):
      result = np.array(result)
      
      if mode == "max":
          pred = result[result[:, 1].argmax()][0]
          if label_map is not None:
              pred = label_map[str(int(pred))]
      elif mode == "n_voting":
          pred = np.bincount(result[:, 0].astype(int)).argmax()
          if label_map is not None:
              pred = label_map[str(int(pred))]
      elif mode == "multi":
          if label_map is not None:
              pred = list(map(lambda a: [label_map[str(int(a[0]))], a[1], a[2], a[3]], result))
          else:
              pred = result.tolist()
    
    return pred


def main(args):
    ######################
    # create model
    ######################
    config = get_config(args.conf)
    model = get_model(config["hparams"]["model"])

    ######################
    # load weights
    ######################
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    
    ######################
    # setup data
    ######################
    if os.path.isdir(args.inp):
        data_list = glob.glob(os.path.join(args.inp, "*.wav"))
    elif os.path.isfile(args.inp):
        data_list = [args.inp]

    ######################
    # run inference
    ######################
    if args.device == "auto":
        device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
    else:
        device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

    model = model.to(f'npu:{NPU_CALCULATE_DEVICE}')
    
    label_map = None
    if args.lmap:
        with open(args.lmap, "r") as f:
            label_map = json.load(f)

    pred_dict = dict()
    for file_path in data_list:
        preds = get_clip_pred(model, file_path, args.wlen, args.stride, args.thresh, config, args.batch_size, device, args.mode, label_map)
        pred_dict[file_path] = preds
    
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "preds_clip.json")

    with open(out_path, "w+") as f:
        json.dump(pred_dict, f)

    print(f"Saved preds to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conf", type=str, required=True, help="Path to config file. Will be used only to construct model and process audio.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument("--inp", type=str, required=True, help="Path to input. Can be a path to a .wav file, or a path to a folder containing .wav files.")
    parser.add_argument("--out", type=str, default="./", help="Path to output folder. Predictions will be stored in {out}/preds.json.")
    parser.add_argument("--lmap", type=str, default=None, help="Path to label_map.json. If not provided, will save predictions as class indices instead of class names.")
    parser.add_argument("--device", type=str, default="auto", help="One of auto, cpu, or cuda.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for batch inference.")
    parser.add_argument("--wlen", type=float, default=1.0, help="Window length. E.g. for wlen = 1, will make inference on 1s windows from the clip.")
    parser.add_argument("--stride", type=float, default=0.2, help="By how much the sliding window will be shifted.")
    parser.add_argument("--thresh", type=float, default=0.85, help="Confidence threshold above which preds will be counted.")
    parser.add_argument("--mode", type=str, default="multi", help="""Prediction logic. One of: max, n_voting, multi.
        -'max' simply checks the confidences of every predicted window in a clip and returns the most confident prediction as the output.
        -'n_voting' returns the most frequent predicted class above the threshold.
        -'multi' expects that there are multiple different keyword classes in the audio. For each audio, the output is a list of lists,
            each sub-list being of the form [class, confidence, start, end].""")
    
    args = parser.parse_args()

    assert os.path.exists(args.inp), f"Could not find input {args.inp}"

    main(args)