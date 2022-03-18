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
import sys
from argparse import ArgumentParser

import torch
import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam, SGD
import numpy as np
from dataset.dataloader_classifier import Dataset
from dataset.config import Config
from models.transformer import Transformer
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def go(arg):
    config = Config()
    train_file = 'sample-data/200410_train_stratshuf_english.csv'
    test_file = 'sample-data/200410_test_stratshuf_chinese_200410_english.csv'
    dataset = Dataset(config)
    dataset.load_data(train_file, test_file)

    NUM_CLS = 338 # taken from the bert code
    vocab_size = len(dataset.vocab)
    # create the model
    model = Transformer(k=arg.dim_model, heads=arg.num_heads, depth=arg.depth,
                        num_tokens=vocab_size, num_classes=NUM_CLS)
    use_cuda = torch.npu.is_available() and not arg.cpu
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

    model = model.to(f'npu:{NPU_CALCULATE_DEVICE}')

    # opt = Adam(params=model.parameters(), lr=arg.lr, betas=(0.9, 0.999), eps=1e-8,
    #                  weight_decay=0, amsgrad=False)
    opt = SGD(params=model.parameters(), lr=arg.lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / config.batch_size), 1.0))

    # training loop
    from torch.utils.tensorboard import SummaryWriter
    # default `log_dir` is "runs" - we'll be more specific here
    tbw = SummaryWriter('runs/experiment_1')

    for e in range(arg.num_epochs):
        print(f'\n epoch {e}')
        model.train(True)
        total_loss = 0.0
        seen = 0
        for batch in tqdm.tqdm(dataset.train_iterator):
            opt.zero_grad()
            if torch.npu.is_available():
                x = batch.text.npu()
                y = (batch.label).type(torch.npu.LongTensor)
            else:
                x = batch.text
                y = (batch.label).type(torch.LongTensor)

            out = model(x.permute(1,0))
            # print(y.size())
            # print(out.size())
            loss = F.nll_loss(out, y)
            total_loss += loss.data

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)
            opt.step()
            sch.step(epoch=e)
            seen += x.size(0)
        tbw.add_scalar('classification/train-loss', e, total_loss/seen)
        print('classification/train-loss:', total_loss/seen)

        with torch.no_grad():

            model.train(False)
            all_preds = []
            all_y = []
            for batch in dataset.val_iterator:
                if torch.npu.is_available():
                    x = batch.text.npu()
                    y = batch.label
                else:
                    x = batch.text
                    y = batch.label

                out = model(x.permute(1, 0))
                # print(out.size())
                predicted = out.argmax(dim=1)
                # print(predicted.size())
                all_preds.extend(predicted.numpy())
                all_y.extend(batch.label.numpy())
                score = accuracy_score(all_y, np.array(all_preds).flatten())

            print(f'-- {"test" if arg.final else "validation"} accuracy {score}')
            # tbw.add_scalar('classification/test-loss', float(loss.item()), e)
    for batch in dataset.test_iterator:
        input = batch.text[0]
        label = batch.label - 1
        print(input)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.01, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--cpu", dest="cpu",
                        help="Use CPU computation.).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--dim_model", dest="dim_model",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
