# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
from easydict import EasyDict as edict
import yaml
import time
import os
import torch
import torch_aie
from tqdm import tqdm
from torch.utils.data import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets
from lib.utils.utils import model_info


def parse_arg():
    parser = argparse.ArgumentParser(description="eval crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    parser.add_argument('--batch_size', help='batch size', default=1, required=True, type=int)
    parser.add_argument('--model_path', help='path to compiled aie model', required=True, type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args


def main():
    # load config
    config, args = parse_arg()

    # set device
    torch_aie.set_device(0)

    #load model
    model = torch.jit.load(args.model_path)
    model.eval()

    #prepare for eval
    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    n_correct = 0
    eval_step = 0
    inf_stream = torch_aie.npu.Stream("npu:0")
    inf_time = []
    print("Ready to infer on dataset.")

    # eval on dataset
    with torch.no_grad():
        for i, (inp, idx) in tqdm(enumerate(val_loader), total=len(val_loader)):
            eval_step += 1
            labels = utils.get_batch_label(val_dataset, idx)
            inp_npu = inp.to("npu:0")

            # inference
            inf_s = time.time()
            with torch_aie.npu.stream(inf_stream):
                preds = model(inp_npu)
            inf_stream.synchronize()
            inf_e = time.time()
            inf_time.append(inf_e - inf_s)

            # checkout result
            preds = preds.to("cpu")
            preds_size = torch.IntTensor([preds.size(0)] * args.batch_size)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    n_correct += 1

    print("Infer on dataset done, calculating results.")
    num_test_sample = eval_step * args.batch_size
    print("[#correct:{} / #total:{}]".format(n_correct, num_test_sample))
    accuracy = n_correct / float(num_test_sample)
    print('Accuray on val set is: {:.4f}'.format(accuracy))

    avg_inf_time = sum(inf_time[3:]) / len(inf_time[3:])
    throughput = args.batch_size / avg_inf_time
    print('Throughput on val set is: {:.2f}'.format(throughput))


if __name__ == '__main__':

    main()