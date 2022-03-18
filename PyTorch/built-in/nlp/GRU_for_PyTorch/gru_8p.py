# Copyright [yyyy] [name of copyright owner]
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

import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import numpy as np
import math
import argparse
import os
import random
import time
import warnings
import en_core_web_sm
import de_core_news_sm

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from torchtext.data import Iterator, BucketIterator
import torch.multiprocessing as mp
import torch.distributed as dist
from apex import amp

from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq

# dataset
spacy_en = en_core_web_sm.load()
spacy_de = de_core_news_sm.load()

# hyperparameter
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
CLIP = 1

MAX = 2147483647


def gen_seeds(num):
    return torch.randint(1, MAX, size=(num,), dtype=torch.float)


seed_init = 0

parser = argparse.ArgumentParser(description='PyTorch Seq2seq-GRU Training')

parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini- size (default: 256), this is the total '
                         ' size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default:1)')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--device', default='npu', type=str,
                    help='npu or gpu')
parser.add_argument('--data-dir', default='/npu/traindata/gru', type=str,
                    help='gru')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--npu', default=0, type=int,
                    help='NPU id to use.')

parser.add_argument('--addr', default='10.136.181.115', type=str,
                    help='master addr')
parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
# apex
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=32., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--ckptpath', default='./seq2seq-gru-model.pth.tar', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--bleu-npu', default=0, type=int,
                    help='NPU id to use.')


def main():
    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        SEED = args.seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29688'

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.process_device_map = device_id_to_process_device_map(args.device_list)

    ngpus_per_node = len(args.process_device_map)

    print('{} node found.'.format(ngpus_per_node))
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        # args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # print('mp.spawn')
        # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        # else:
        # Simply call main_worker function
        args.world_size = ngpus_per_node * args.world_size
        print("---------------args.npu", args.npu)
        main_worker(args.npu, ngpus_per_node, args)


def main_worker(npu, ngpus_per_node, args):
    args.npu = args.process_device_map[npu]
    print('---------------args.npu', args.npu)

    CALCULATE_DEVICE = "npu:{}".format(args.npu)
    torch.npu.set_device(CALCULATE_DEVICE)

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.npu

            dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)

    # parpare dataset
    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True, fix_length=46)
    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True, fix_length=46)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),root=args.data_dir,
                                                        fields=(SRC, TRG))
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    device = CALCULATE_DEVICE

    train_iterator = Iterator(train_data, batch_size=args.batch_size, device=device)
    valid_iterator = Iterator(valid_data, batch_size=args.batch_size, device=device)
    test_iterator = Iterator(test_data, batch_size=args.batch_size, device=device)

    seed_init = gen_seeds(32 * 1024 * 12).float().to(CALCULATE_DEVICE)

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT, seed=seed_init).to(CALCULATE_DEVICE)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, seed=seed_init).to(CALCULATE_DEVICE)

    model = Seq2Seq(enc, dec, device)
    model.apply(init_weights)
    model = model.to(CALCULATE_DEVICE)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.npu], broadcast_buffers=False)

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX).to(CALCULATE_DEVICE)
    best_valid_loss = float('inf')

    for epoch in range(args.epochs):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, epoch, args, ngpus_per_node, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion, args, ngpus_per_node, CLIP)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'seq2seq-gru-model.pth.tar')

            print(f'Epoch：{epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t  Val Loss: {valid_loss:.3f} |   Val PPL: {math.exp(valid_loss):7.3f}')


def train(model, train_iterator, optimizer, criterion, epoch, args, ngpus_per_node, clip):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_iterator),
                             [batch_time, data_time, losses],
                             prefix="Epoch: [{}]".format(epoch))

    epoch_loss = 0
    CALCULATE_DEVICE = "npu:{}".format(args.npu)
    print('train_CALCULATE_DEVICE', CALCULATE_DEVICE)
    model.train()

    end = time.time()
    for i, batch in enumerate(train_iterator):
        if i == len(train_iterator) - 1:
            continue

        data_time.update(time.time() - end)
        src = batch.src.to(CALCULATE_DEVICE)
        trg = batch.trg.to(CALCULATE_DEVICE)

        optimizer.zero_grad()

        output = model(src, trg).to(CALCULATE_DEVICE)

        output_dim = output.shape[-1]

        trg = trg.to(torch.int32)

        output = output[1:].view(-1, output_dim).to(CALCULATE_DEVICE)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        losses.update(loss.item())
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        batch_time.update(time.time() - end)
        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                progress.display(i)

        epoch_loss += loss.item()
        end = time.time()

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        print("[npu id:", args.npu, "]",
              '* FPS@all {:.3f}'.format(ngpus_per_node * args.batch_size / batch_time.avg))

    return epoch_loss / len(train_iterator)


def evaluate(model, valid_iterator, criterion, args, ngpus_per_node, clip):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(valid_iterator),
        [batch_time, losses],
        prefix="valid:")

    model.eval()
    CALCULATE_DEVICE = "npu:{}".format(args.npu)
    epoch_loss = 0

    with torch.no_grad():
        end = time.time()

        for i, batch in enumerate(valid_iterator):
            src = batch.src.to(CALCULATE_DEVICE)
            trg = batch.trg.to(CALCULATE_DEVICE)

            output = model(src, trg, 0).to(CALCULATE_DEVICE)  # turn off teacher forcing

            trg = trg.to(torch.int32)
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            losses.update(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    progress.display(i)

            epoch_loss += loss.item()

    return epoch_loss / len(valid_iterator)


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()

    # tokenize input
    if isinstance(sentence, str):
        nlp = de_core_news_sm.load()
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    # add <sos> and <eos>
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    # get input's one-hot vec
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    # add a  dim and convert into tensor
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor)

    hidden = encoder_outputs

    # get first decoder input (<sos>)'s one hot
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:]


def calculate_bleu(data, src_field, trg_field, model, device, max_len=46):
    trgs = []
    pred_trgs = []

    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg = translate_sentence(src, src_field, trg_field, model, device, max_len)

        # cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = 5

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        if self.count > (self.start_count_index * n):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * n)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("[npu id:", '0', "]", '\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()
    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id
    return process_device_map


if __name__ == "__main__":
    main()
