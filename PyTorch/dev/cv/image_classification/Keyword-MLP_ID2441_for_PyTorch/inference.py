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
"""Run inference on short ~1s clips, like the ones in the Speech Commands dataset."""

from argparse import ArgumentParser
from config_parser import get_config
import torch
from torch.utils.data import DataLoader
from utils.misc import get_model
from utils.dataset import GoogleSpeechDataset
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


@torch.no_grad()
def get_preds(net, dataloader, device) -> list:
    """Performs inference."""

    net.eval()
    preds_list = []

    for data in tqdm(dataloader):
        data  = data.to(f'npu:{NPU_CALCULATE_DEVICE}')
        out = net(data)
        preds = out.argmax(1).cpu().numpy().ravel().tolist()
        preds_list.extend(preds)

    return preds_list


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

    dataset = GoogleSpeechDataset(
        data_list=data_list,
        label_map=None,
        audio_settings=config["hparams"]["audio"],
        aug_settings=None,
        cache=0
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    ######################
    # run inference
    ######################
    if args.device == "auto":
        device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
    else:
        device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

    model = model.to(f'npu:{NPU_CALCULATE_DEVICE}')
    preds = get_preds(model, dataloader, device)

    ######################
    # save predictions
    ######################
    if args.lmap:
        with open(args.lmap, "r") as f:
            label_map = json.load(f)
        preds = list(map(lambda a: label_map[str(a)], preds))
    
    pred_dict = dict(zip(data_list, preds))
    
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "preds.json")

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
    parser.add_argument('--apex', action='store_true',help='Use apex for mixed precision training')
parser.add_argument('--apex-opt-level',default='O2',type=str,help='For apex mixed precision training, O0 for FP32 training, O1 for mixed precision training')
    
    args = parser.parse_args()

    assert os.path.exists(args.inp), f"Could not find input {args.inp}"

    main(args)