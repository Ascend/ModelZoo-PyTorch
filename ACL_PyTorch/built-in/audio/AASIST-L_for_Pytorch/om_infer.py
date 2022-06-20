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

import argparse
import json
import os
from importlib import import_module
from pathlib import Path
from shutil import copy
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data_utils import Dataset_ASVspoof2019_devNeval, genSpoof_list
from evaluation import calculate_tDCF_EER

import acl
from pyacl.acl_infer import AclNet, init_acl, release_acl


def main(args):
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"

    # define related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    eval_trial_path = (database_path/"ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(track, prefix_2019))
    eval_score_path = output_dir/"eval_scores.txt"

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    # define dataloaders
    eval_loader = get_loader(database_path, config, args)
    
    # load model
    init_acl(args.device_id)
    model = AclNet(device_id=args.device_id, model_path=args.om_path)

    # evaluates pretrained model and exit script
    print("Start evaluation...")
    produce_evaluation_file(eval_loader, model, eval_score_path, eval_trial_path)
    calculate_tDCF_EER(cm_scores_file=eval_score_path,
                        asv_score_file=database_path /
                        config["asv_score_path"],
                        output_file=output_dir / "t-DCF_EER.txt")

    del model
    release_acl(args.device_id)


def get_loader(database_path, config, args):
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    eval_database_path = database_path/"ASVspoof2019_{}_eval/".format(track)
    eval_trial_path = (database_path/"ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(track, prefix_2019))

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return eval_loader


def produce_evaluation_file(data_loader, model, save_path, trial_path):
    """Perform evaluation and save the score to a file"""
    
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.numpy().astype(np.float32)
        _, batch_out = model(batch_x)[0]
        batch_score = (batch_out[:, 1]).ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config", default='./config/AASIST-L.conf', type=str, help="configuration file")
    parser.add_argument('--om-path', default="aasist_bs1.om", type=str, help='path to the om model')
    parser.add_argument('--batch-size', default=1, type=int, help='batch size')
    parser.add_argument("--output_dir", default="./output", type=str, help="output directory for results")
    parser.add_argument('--device_id', default=0, type=int, help='device id')

    main(parser.parse_args())
