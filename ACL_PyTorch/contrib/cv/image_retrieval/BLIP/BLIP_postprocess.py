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

# coding:utf-8


import functools
import os
import numpy as np
import json
import utils as utils
import torch
from models.blip_itm import blip_itm
import re
import argparse


@torch.no_grad()
def evaluation(image_bin_path, image_feat_bin_path, text_bin_path, ids_path, mask_path, model):
    k_test = 20

    print("k_test: ", k_test)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    device = torch.device('cpu')
    print("Running on : ", device)

    model.to(device)
    model.eval()

    # make sure right orders of bin files according to the sumary.json file
    def resolve_summary_file(summary_file):
        sumary = json.load(open(summary_file, 'r'))
        bin_dir = ['' for i in range(len(sumary['filesinfo']))]
        for key in sumary['filesinfo']:
            infile = sumary['filesinfo'][key]['infiles'][0]
            id = int(infile.split('/')[-1][0:-4])
            outfile = sumary['filesinfo'][key]['outfiles'][0]
            outfile = outfile.split('/')[-1]
            bin_dir[id] = outfile
        return bin_dir

    image_embed_sumary_file = os.path.join(image_bin_path, 'sumary.json')
    image_embed_dir = resolve_summary_file(image_embed_sumary_file)

    image_feat_summary_file = os.path.join(image_feat_bin_path, 'sumary.json')
    image_feat_dir = resolve_summary_file(image_feat_summary_file)

    text_embed_sumary_file = os.path.join(text_bin_path, 'sumary.json')
    text_embed_dir = resolve_summary_file(text_embed_sumary_file)

    # read bin files
    text_embeds = []  # [256]
    image_embeds = []  # [256]
    image_feats = []  # [577*768]

    for image_bin in image_embed_dir:
        files = np.fromfile(image_bin_path + image_bin,
                            dtype=np.float32)  # [256]
        image_embeds.append(files)

    for image_feat_bin in image_feat_dir:
        files = np.fromfile(image_feat_bin_path +
                            image_feat_bin, dtype=np.float32)  # [443136]
        data = files.reshape(577, 768)
        image_feats.append(data)

    for text_bin in text_embed_dir:
        text_files = np.fromfile(
            text_bin_path + text_bin, dtype=np.float32)  # [256]
        text_embeds.append(text_files)

    print("Load bins completed.")

    image_embeds = torch.tensor(np.array(image_embeds))
    image_feats = torch.tensor(np.array(image_feats))
    text_embeds = torch.tensor(np.array(text_embeds))

    print("convert bins to tensors completed")

    text_ids = []
    text_atts = []

    #  make sure the orders of bin files : [0.bin, 1.bin, 2.bin, 3.bin, ... , 4998.bin, 4999.bin]
    def compare_str(s1, s2):
        if len(s1) < len(s2):
            return -1
        elif len(s1) > len(s2):
            return 1
        elif s1 < s2:
            return -1
        elif s1 > s2:
            return 1
        else:
            return 0

    ids_dir = os.listdir(ids_path)
    ids_dir.sort(key=functools.cmp_to_key(compare_str))
    mask_dir = os.listdir(mask_path)
    mask_dir.sort(key=functools.cmp_to_key(compare_str))

    for ids_bin in ids_dir:
        if 'sumary.json' == ids_bin:
            continue
        files = np.fromfile(ids_path + ids_bin, dtype=np.int64)
        text_ids.append(files)

    for atts_bin in mask_dir:
        if 'sumary.json' == atts_bin:
            continue
        files = np.fromfile(mask_path + atts_bin, dtype=np.int64)
        text_atts.append(files)

    print("load text ids and mask completed.")

    text_ids = torch.tensor(np.array(text_ids))
    text_atts = torch.tensor(np.array(text_atts))
    text_ids[:, 0] = model.tokenizer.enc_token_id

    print("convert text ids and masks to tensor completed.")

    print("text_ids, text_atts:", text_ids.shape, text_atts.shape)
    text_ids = text_ids.to(torch.int64)

    sims_matrix = image_embeds @ text_embeds.t()
    sims_matrix = sims_matrix.to(device)
    score_matrix_i2t = torch.full(
        (len(image_embed_dir), len(text_embed_dir)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0), start+step)

    print("### begin calcute score_matrix_i2t")

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):

        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_feats[start+i].repeat(k_test, 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[
                                 :-1], dtype=torch.long).to(device)

        output = model.text_encoder(text_ids[topk_idx].to(device),
                                    attention_mask=text_atts[topk_idx].to(
                                        device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]

        score_matrix_i2t[start+i, topk_idx] = score + topk_sim

    print("### end calcute score_matrix_i2t")

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(text_embed_dir), len(image_embed_dir)), -100.0)
    score_matrix_t2i = score_matrix_t2i.to(device)

    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0), start+step)

    print("### begin calcute score_matrix_t2i")

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[
                                 :-1], dtype=torch.long).to(device)

        output = model.text_encoder(text_ids[start+i].repeat(k_test, 1).to(device),
                                    attention_mask=text_atts[start +
                                                             i].repeat(k_test, 1).to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start+i, topk_idx] = score + topk_sim

    print("### end calcute score_matrix_t2i")

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


def itm_eval(scores_i2t, scores_t2i, text2image, image2text):
    print("begin calcute matrics.")
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in image2text[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == text2image[index])[0][0]
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    eval_result = {'text_r1': tr1,
                   'text_r5': tr5,
                   'text_r10': tr10,
                   'text_r_mean': tr_mean,
                   'image_r1': ir1,
                   'image_r5': ir5,
                   'image_r10': ir10,
                   'image_r_mean': ir_mean,
                   'r_mean': r_mean}
    print('eval_result:', eval_result)
    return eval_result


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_embed_path',
                        default='./coco2014_infer/text_embed/xxx/')
    parser.add_argument('--image_embed_path',
                        default='./coco2014_infer/image_embed/xxx/')
    parser.add_argument('--image_feat_path',
                        default='./coco2014_infer/image_feat/xxx/')
    parser.add_argument('--coco_bin_path', default='./coco2014_bin/')
    parser.add_argument('--pth_path', default='/model_base_retrieval_coco.pth')
    args = parser.parse_args()

    print("running calculate metrics...")

    image_embed_path = args.image_embed_path
    text_embed_path = args.text_embed_path
    image_feat_path = args.image_feat_path

    ids_path = os.path.join(args.coco_bin_path, 'ids/')
    mask_path = os.path.join(args.coco_bin_path, 'mask/')

    pretrained_model = args.pth_path

    model = blip_itm(pretrained=pretrained_model)
    with torch.no_grad():
        score_matrix_i2t, score_matrix_t2i = evaluation(
            image_embed_path, image_feat_path, text_embed_path, ids_path, mask_path, model)

    ann_root = 'annotation'
    # 获取测试集
    filenames = {'val': 'coco_karpathy_val.json',
                 'test': 'coco_karpathy_test.json'}
    annotation = json.load(
        open(os.path.join(ann_root, filenames['test']), 'r'))

    text = []
    image = []
    text2image = {}
    image2text = {}

    text_id = 0
    for image_id, ann in enumerate(annotation):
        image.append(ann['image'])
        image2text[image_id] = []
        for i, caption in enumerate(ann['caption']):
            text.append(pre_caption(caption, 30))
            image2text[image_id].append(text_id)
            text2image[text_id] = image_id
            text_id += 1

    print("prepare test data completed.")
    itm_eval(score_matrix_i2t, score_matrix_t2i, text2image, image2text)
