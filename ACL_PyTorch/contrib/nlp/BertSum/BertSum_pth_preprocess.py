# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import argparse
import os
import torch
from tqdm import tqdm

sys.path.append("./BertSum/src")
from models import data_loader
from models.data_loader import load_dataset
from models.trainer import build_trainer


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def preprocess(args, device, max_shape_1=512, max_shape_2=37):
    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.batch_size, device,
                                       shuffle=False, is_test=True)

    input_names = ["src", "segs", "clss", "mask", "mask_cls"]
    for input_name in input_names:
        input_path = os.path.join(args.out_path, input_name)
        os.makedirs(input_path, exist_ok=True)

    #############################above get max dimension ###########################
    def save_batch_data(data_idx, batch_data):
        input_data = [batch_data.src, batch_data.segs, batch_data.clss,
                      batch_data.mask, batch_data.mask_cls]
        for batch_idx in range(batch_data.src.shape[0]):
            for input_idx, input_name in enumerate(input_names):
                input_data[input_idx][batch_idx].numpy().tofile(os.path.join(
                    args.out_path, input_name, f"data_{data_idx}_{batch_idx}.bin"
                ))

    for data_idx, batch in tqdm(enumerate(test_iter)):
        if batch.src[0].shape[0] < max_shape_1:
            add_zero = (torch.zeros([batch.src.shape[0],
                                     max_shape_1 - batch.src[0].shape[0]])).long()
            add_bool = torch.zeros([batch.src.shape[0],
                                    max_shape_1-batch.src[0].shape[0]], dtype=torch.bool)
            batch.src = torch.cat([batch.src, add_zero], dim=1)
            batch.segs = torch.cat([batch.segs, add_zero], dim=1)
            batch.mask = torch.cat([batch.mask, add_bool], dim=1)
        if batch.clss[0].shape[0] < max_shape_2:
            add_zero = (torch.zeros([batch.clss.shape[0],
                                     max_shape_2-batch.clss[0].shape[0]])).long()
            add_bool = torch.zeros([batch.clss.shape[0], max_shape_2-batch.clss[0].shape[0]], dtype=torch.bool)
            batch.clss = torch.cat([batch.clss, add_zero], dim=1)
            batch.mask_cls = torch.cat([batch.mask_cls, add_bool], dim=1)
        save_batch_data(data_idx, batch)


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder", default='classifier', type=str, choices=['classifier', 'transformer', 'rnn', 'baseline'])
    parser.add_argument("-mode", default='test', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='./bert_data')
    parser.add_argument("-model_path", default='./models/')
    parser.add_argument("-result_path", default='./results/cnndm')
    parser.add_argument("-temp_dir", default='./temp')
    parser.add_argument("-bert_config_path", default='BertSum/bert_config_uncased_base.json')

    parser.add_argument("-batch_size", default=600, type=int)

    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-hidden_size", default=128, type=int)
    parser.add_argument("-ff_size", default=512, type=int)
    parser.add_argument("-heads", default=4, type=int)
    parser.add_argument("-inter_layers", default=2, type=int)
    parser.add_argument("-rnn_size", default=512, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default='', type=str)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-world_size", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='./preprocess_cnndm.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-out_path", default="")

    args = parser.parse_args()
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = -1 if device == "cpu" else 0

    preprocess(args, device)
