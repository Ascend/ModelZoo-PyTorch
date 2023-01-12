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

import torch
import os
import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm
from pytorch_pretrained_bert import BertConfig
sys.path.append("./BertSum/src")
from models.model_builder import Summarizer
from models import data_loader
from models.data_loader import load_dataset
from models.stats import Statistics
from others.utils import test_rouge
from others.logging import logger


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100
    )


def pre_postprocess(args):
    file_names = os.listdir(args.result_dir)
    data_info = {}
    for file_name in sorted(file_names):
        _, data_idx, batch_idx, suffix = file_name.split("_")
        if data_idx not in data_info:
            data_info[data_idx] = {}
        if data_info[data_idx].get(batch_idx) is None:
            data_info[data_idx][batch_idx] = []
        data_info[data_idx][batch_idx].append(suffix)

    summary_outs = []
    summary_scores = []
    for data_idx in tqdm(data_info):
        outputs = []
        sent_scores = []
        for batch_idx in data_info[data_idx]:
            prefix = f'data_{data_idx}_{batch_idx}'
            score = np.fromfile(os.path.join(args.result_dir, f'{prefix}_0.bin'),
                                dtype="float32")
            output = np.fromfile(os.path.join(args.result_dir, f'{prefix}_1.bin'),
                                 dtype="bool")
            outputs.append(output)
            sent_scores.append(score)
        summary_outs.append(torch.tensor(outputs))
        summary_scores.append(torch.tensor(sent_scores))

    test_iter = data_loader.Dataloader(args,
                                       load_dataset(args, 'test', shuffle=False),
                                       args.batch_size, device,
                                       shuffle=False, is_test=True)
    for data_idx, batch in enumerate(test_iter):
        labels = batch.labels
        summary_outs[data_idx] = summary_outs[data_idx][:, 0:labels.shape[1]]
        summary_scores[data_idx] = summary_scores[data_idx][:, 0:labels.shape[1]]

    return summary_scores, summary_outs


def test(args, step, device, cal_lead=False, cal_oracle=False):
        test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        stats = Statistics()
        can_path = '%s_step%d.candidate' % (args.result_path, step)
        gold_path = '%s_step%d.gold' % (args.result_path, step)
        
        sent,output = pre_postprocess(args)
        Loss = torch.nn.BCELoss(reduction='none')
        sum = 0
        k = 0
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        labels = batch.labels
                        
                        gold = []
                        pred = []
                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        elif (cal_oracle):
                            selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                            range(batch.batch_size)]
                        else:
                            if labels.shape[0] != sent[k].shape[0]:
                                #labels = labels[sent[k].shape[0],:]
                                k = k + 1
                                sum = sum + 1
                                continue
                                
                            loss = Loss(sent[k], labels.float())
                            
                            if loss.shape[1] != output[k].shape[1]:
                                k = k + 1
                                continue
                            
                            loss = (loss * output[k].float()).sum()
                            batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                            stats.update(batch_stats)
        
                            sent_scores = sent[k] + output[k].float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1)
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if(len(batch.src_str[i]) == 0):
                                continue
                            for j in selected_ids[i][:len(batch.src_str[i])]:
                                if(j >= len(batch.src_str[i])):
                                    continue
                                candidate = batch.src_str[i][j].strip()
                                if(args.block_trigram):
                                    if(not _block_tri(candidate, _pred)):
                                        _pred.append(candidate)
                                else:
                                    _pred.append(candidate)

                                if ((not cal_oracle) and (not args.recall_eval) and len(_pred) == 3):
                                    break

                            _pred = '<q>'.join(_pred)
                            if(args.recall_eval):
                                _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                            pred.append(_pred)
                            gold.append(batch.tgt_str[i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip()+'\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip()+'\n')
                        k = k + 1
        if(step != -1 and args.report_rouge):
            print(can_path)
            print(gold_path)
            rouges = test_rouge(args.temp_dir, can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))

        return stats


if __name__ == "__main__":
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
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-result_dir", default="")

    args = parser.parse_args()
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = -1 if device == "cpu" else 0
    test(args,0,device)
