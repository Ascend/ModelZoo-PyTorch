# Copyright 2023 Huawei Technologies Co., Ltd
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
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data_utils import Dataset_ASVspoof2019_devNeval, genSpoof_list
from evaluation import calculate_tDCF_EER

from ais_bench.infer.interface import InferSession


def main(args):
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"

    # define related paths
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    eval_trial_path = (database_path / "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(track, prefix_2019))
    eval_score_path = "eval_scores.txt"

    # define dataloaders
    eval_loader = get_loader(database_path, config, args)

    # load model
    model = InferSession(args.device_id, args.om_path)

    # evaluates pretrained model and exit script
    print("Start evaluation...")
    produce_evaluation_file(eval_loader, model, eval_score_path, eval_trial_path)
    calculate_tDCF_EER(cm_scores_file=eval_score_path,
                       asv_score_file=database_path /
                                      config["asv_score_path"],
                       output_file="t-DCF_EER.txt")


def get_loader(database_path, config, args):
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)
    eval_trial_path = (database_path / "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(track, prefix_2019))

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
        # handle the case where the last batch is not divisible
        infer_batch = batch_x.shape[0]
        padding = False
        batch_size = model.get_inputs()[0].shape[0]
        if infer_batch != batch_size:
            batch_x = np.pad(batch_x, ((0, batch_size - infer_batch), (0, 0)), 'constant', constant_values=0)
            padding = True
        else:
            batch_x = batch_x.numpy().astype(np.float32)

        batch_out = model.infer([batch_x])[1]
        if padding == True:
            batch_out = batch_out[:infer_batch]

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
    parser.add_argument('--device_id', default=0, type=int, help='device id')

    main(parser.parse_args())
