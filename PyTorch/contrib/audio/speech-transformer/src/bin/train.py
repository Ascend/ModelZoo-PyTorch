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
import argparse
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import os
from collections import OrderedDict
from data import AudioDataLoader, AudioDataset
from decoder import Decoder
from encoder import Encoder
from transformer import Transformer
from solver import Solver
from utils import process_dict
from optimizer import TransformerOptimizer
from apex import amp
import sys
import random
import numpy as np
parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Training "
    "(Transformer framework).")
# General config
# Task related
parser.add_argument('--train-json', type=str, default=None,
                    help='Filename of train label data (json)')
parser.add_argument('--valid-json', type=str, default=None,
                    help='Filename of validation label data (json)')
parser.add_argument('--dict', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')
# Low Frame Rate (stacking and skipping frames)
parser.add_argument('--LFR_m', default=4, type=int,
                    help='Low Frame Rate: number of frames to stack')
parser.add_argument('--LFR_n', default=3, type=int,
                    help='Low Frame Rate: number of frames to skip')
# Network architecture
# encoder
# TODO: automatically infer input dim
parser.add_argument('--d_input', default=80, type=int,
                    help='Dim of encoder input (before LFR)')
parser.add_argument('--n_layers_enc', default=6, type=int,
                    help='Number of encoder stacks')
parser.add_argument('--n_head', default=8, type=int,
                    help='Number of Multi Head Attention (MHA)')
parser.add_argument('--d_k', default=64, type=int,
                    help='Dimension of key')
parser.add_argument('--d_v', default=64, type=int,
                    help='Dimension of value')
parser.add_argument('--d_model', default=512, type=int,
                    help='Dimension of model')
parser.add_argument('--d_inner', default=2048, type=int,
                    help='Dimension of inner')
parser.add_argument('--dropout', default=0.1, type=float,
                    help='Dropout rate')
parser.add_argument('--pe_maxlen', default=5000, type=int,
                    help='Positional Encoding max len')
# decoder
parser.add_argument('--d_word_vec', default=512, type=int,
                    help='Dim of decoder embedding')
parser.add_argument('--n_layers_dec', default=6, type=int,
                    help='Number of decoder stacks')
parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int,
                    help='share decoder embedding with decoder projection')
# Loss
parser.add_argument('--label_smoothing', default=0.1, type=float,
                    help='label smoothing')

# Training config
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch-size', default=32, type=int,
                    help='Batch size')
parser.add_argument('--batch_frames', default=0, type=int,
                    help='Batch frames. If this is not 0, batch size will make no sense')
parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                    help='Batch size is reduced if the input sequence length > ML')
parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                    help='Batch size is reduced if the output sequence length > ML')
parser.add_argument('--num-workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--k', default=1.0, type=float,
                    help='tunable scalar multiply to learning rate')
parser.add_argument('--warmup_steps', default=4000, type=int,
                    help='warmup steps')
# save and load model
parser.add_argument('--save-folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue-from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model-path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print-freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_lr', dest='visdom_lr', type=int, default=0,
                    help='Turn on visdom graphing learning rate')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom-id', default='Transformer training',
                    help='Identifier for visdom run')

parser.add_argument('--is_distributed', default=False,
                    type=bool)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--is_npu', default=True, type=bool)
parser.add_argument('--num_of_gpus', default=8, type=int)
parser.add_argument('--world_size',default=8, type = int)
parser.add_argument('--val_batch_size', default=64, type=int,
                    help='Val batch size')
parser.add_argument('--no-bin', default=False, action='store_true')
                    help='identifier to enable binary mode')

IS_DISTRIBUTED = False

def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if(k[0: 7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def load_state(args,model, optimizer):
    # Reset
    package = None
    if args.continue_from:
        print('Loading checkpoint model %s' % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
        package['state_dict'] = proc_node_module(package, 'state_dict')
        package['optim_dict'] = proc_node_module(package, 'optim_dict')
        model.load_state_dict(package['state_dict'])
        optimizer.load_state_dict(package['optim_dict'])
    return package


def seed_everything():
    random.seed(1234)
    os.environ['PYTHONHASHSEED'] = str(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

def main(args):
    # Construct Solver
    # data
    IS_DISTRIBUTED = bool(args.is_distributed)
    tr_dataset = AudioDataset(args.train_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out,
                              batch_frames=args.batch_frames)
    cv_dataset = AudioDataset(args.valid_json, args.val_batch_size,
                              args.maxlen_in, args.maxlen_out,
                              batch_frames=args.batch_frames)
    if IS_DISTRIBUTED:
        if args.is_npu:
            torch.npu.set_device(args.local_rank)
        else:
            torch.cuda.set_device(args.local_rank)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '23001'
        torch.distributed.init_process_group(backend='hccl',init_method='env://',world_size=args.world_size, rank = int(args.local_rank))
        args.tr_sampler = torch.utils.data.distributed.DistributedSampler(tr_dataset)
        tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                num_workers=args.num_workers,
                                shuffle=False,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n,
                                pin_memory = False,
                                sampler=args.tr_sampler)
        cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=args.num_workers,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n)
    else:
        args.num_of_gpus = 1
        tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                num_workers=args.num_workers,
                                shuffle=args.shuffle,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n)
        cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=args.num_workers,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n)
    # load dictionary and generate char_list, sos_id, eos_id
    char_list, sos_id, eos_id = process_dict(args.dict)
    vocab_size = len(char_list)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    encoder = Encoder(args.d_input * args.LFR_m, args.n_layers_enc, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout, pe_maxlen=args.pe_maxlen)
    decoder = Decoder(sos_id, eos_id, vocab_size,
                      args.d_word_vec, args.n_layers_dec, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout,
                      tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                      pe_maxlen=args.pe_maxlen)
    seed_everything()
    model = Transformer(encoder, decoder)
    if args.is_npu:
        from apex.optimizers.npu_fused_adam import NpuFusedAdam
    optimizer = NpuFusedAdam(model.parameters(), betas=(0.9,0.98), eps=1e-09) if args.is_npu else torch.optim.Adam(model.parameters(), betas=(0.9,0.98), eps =1e-09)
    package = load_state(args, model, optimizer)
    if not package:
        package = {
            # encoder
            'd_input': model.encoder.d_input,
            'n_layers_enc': model.encoder.n_layers,
            'n_head': model.encoder.n_head,
            'd_k': model.encoder.d_k,
            'd_v': model.encoder.d_v,
            'd_model': model.encoder.d_model,
            'd_inner': model.encoder.d_inner,
            'dropout': model.encoder.dropout_rate,
            'pe_maxlen': model.encoder.pe_maxlen,
            # decoder
            'sos_id': model.decoder.sos_id,
            'eos_id': model.decoder.eos_id,
            'vocab_size': model.decoder.n_tgt_vocab,
            'd_word_vec': model.decoder.d_word_vec,
            'n_layers_dec': model.decoder.n_layers,
            'tgt_emb_prj_weight_sharing': model.decoder.tgt_emb_prj_weight_sharing,
        }
    # optimizer
    if args.is_npu:
        model.npu()
        model, optimizer = amp.initialize(model, 
                            optimizer,
                            opt_level="O2",loss_scale='dynamic', combine_grad=True)
    else:
        model.cuda()
        model, optimizer = amp.initialize(model, 
                            optimizer,
                            opt_level="O2",loss_scale=128.0)
    if IS_DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],broadcast_buffers=False)

    optimizier = TransformerOptimizer(
        optimizer,
        args.k,
        args.d_model,
        args.warmup_steps)

    # solver
    solver = Solver(data, model, optimizier, args, package)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print('{} python interpreter is being used now'.format(sys.executable))
    print(args)
    option={}
    option['ACL_OP_SELECT_IMPL_MODE'] = 'high_performance'
    torch.npu.set_option(option)
    if not args.no_bin:
        torch.npu.set_compile_mode(jit_compile=False)
    main(args)

