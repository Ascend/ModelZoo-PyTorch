# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""argparser configuration"""

import argparse
import os
import torch


def add_model_config_args(parser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')

    group.add_argument('--pretrained-bert', action='store_true',
                       help='use a pretrained bert-large-uncased model instead'
                       'of initializing from scratch. See '
                       '--tokenizer-model-type to specify which pretrained '
                       'BERT model to use')
    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='dropout probability for attention weights')
    group.add_argument('--num-attention-heads', type=int, default=16,
                       help='num of transformer attention heads')
    group.add_argument('--hidden-size', type=int, default=1024,
                       help='tansformer hidden size')
    group.add_argument('--intermediate-size', type=int, default=None,
                       help='transformer embedding dimension for FFN'
                       'set to 4*`--hidden-size` if it is None')
    group.add_argument('--num-layers', type=int, default=24,
                       help='num decoder layers')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='layer norm epsilon')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='dropout probability for hidden state transformer')
    group.add_argument('--max-position-embeddings', type=int, default=512,
                       help='maximum number of position embeddings to use')
    group.add_argument('--vocab-size', type=int, default=30522,
                       help='vocab size to use for non-character-level '
                       'tokenization. This value will only be used when '
                       'creating a tokenizer')
    group.add_argument('--deep-init', action='store_true',
                       help='initialize bert model similar to gpt2 model.'
                       'scales initialization of projection layers by a '
                       'factor of 1/sqrt(2N). Necessary to train bert '
                       'models larger than BERT-Large.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--cpu-optimizer', action='store_true',
                                   help='Run optimizer on CPU')
    group.add_argument('--cpu_torch_adam', action='store_true',
                                   help='Use Torch Adam as optimizer on CPU.')

    return parser


def add_fp16_config_args(parser):
    """Mixed precision arguments."""

    group = parser.add_argument_group('fp16', 'fp16 configurations')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode')
    group.add_argument('--fp32-embedding', action='store_true',
                       help='embedding in fp32')
    group.add_argument('--fp32-layernorm', action='store_true',
                       help='layer norm in fp32')
    group.add_argument('--fp32-tokentypes', action='store_true',
                       help='embedding token types in fp32')
    group.add_argument('--fp32-allreduce', action='store_true',
                       help='all-reduce in fp32')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale')
    group.add_argument('--min-scale', type=float, default=1,
                       help='Minimum loss scale for dynamic loss scale')

    return parser


def add_training_args(parser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    group.add_argument('--do_train', action='store_true',
                       help="Do training")
    group.add_argument('--do_eval', action='store_true',
                       help="Do evaluation")
    group.add_argument('--batch-size', type=int, default=4,
                       help='Data Loader batch size')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='weight decay coefficient for L2 regularization')
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='checkpoint activation to allow for training '
                       'with larger models and sequences')
    group.add_argument('--checkpoint-num-layers', type=int, default=1,
                       help='chunk size (number of layers) for checkpointing')
    group.add_argument('--deepspeed-activation-checkpointing', action='store_true',
                       help='uses activation checkpointing from deepspeed')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--epoch', type=int, default=10,
                       help='total number of iterations to train over all training runs')
    group.add_argument('--log-interval', type=int, default=100,
                       help='report interval')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after this many new iterations.')

    group.add_argument('--seed', type=int, default=1234,
                       help='random seed')
    # Batch prodecuer arguments
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token.')
    
    # Learning rate.
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='learning rate decay function')
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    # model checkpointing
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=5000,
                       help='number of iterations between saves')
    group.add_argument('--no-save-optim', action='store_true',
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true',
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true',
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    # distributed training args
    group.add_argument('--distributed-backend', default='hccl',
                       help='which backend to use for distributed '
                       'training. One of [huawei, hccl]')

    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')

    group.add_argument('--results_dir', type=str, default=None,
                       help='The dir to save the model.')
    group.add_argument('--model_name', type=str, default="test",
                       help="The name you give to the model.")

    # eval
    group.add_argument('--eval_ckpt_path', type=str, default=None,
                       help='The checkpoint path used for evaluation')

    return parser


def add_evaluation_args(parser):
    """Evaluation arguments."""

    group = parser.add_argument_group('validation', 'validation configurations')

    group.add_argument('--eval-batch-size', type=int, default=None,
                       help='Data Loader batch size for evaluation datasets.'
                       'Defaults to `--batch-size`')
    group.add_argument('--eval-iters', type=int, default=100,
                       help='number of iterations to run for evaluation'
                       'validation/test for')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='interval between running evaluation on validation set')
    group.add_argument('--eval-seq-length', type=int, default=None,
                       help='Maximum sequence length to process for '
                       'evaluation. Defaults to `--seq-length`')
    group.add_argument('--eval-max-preds-per-seq', type=int, default=None,
                       help='Maximum number of predictions to use for '
                       'evaluation. Defaults to '
                       'math.ceil(`--eval-seq-length`*.15/10)*10')
    group.add_argument('--overlapping-eval', type=int, default=32,
                       help='sliding window for overlapping eval ')
    group.add_argument('--cloze-eval', action='store_true',
                       help='Evaluation dataset from `--valid-data` is a cloze task')
    group.add_argument('--eval-hf', action='store_true',
                       help='perform evaluation with huggingface openai model.'
                       'use `--load` to specify weights path to be loaded')
    group.add_argument('--load-openai', action='store_true',
                       help='load openai weights into our model. Use `--load` '
                       'to specify weights path to be loaded')

    return parser

def add_text_generate_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('Text generation', 'configurations')
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--top_p", type=float, default=0.0)
    group.add_argument("--top_k", type=int, default=0)
    group.add_argument("--out-seq-length", type=int, default=256)
    return parser


def add_data_args(parser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument('--data_dir', type=str, required=True,
                       help="Training data dir")
    group.add_argument('--mmap-warmup', action='store_true',
                       help='Warm up mmap files.')
    group.add_argument('--model-parallel-size', type=int, default=1,
                       help='size of the model parallel.')
    group.add_argument('--shuffle', action='store_true',
                       help='Shuffle data. Shuffling is deterministic '
                       'based on seed and current epoch.')
    group.add_argument('--use-npy-data-loader', action='store_true',
                       help='Use the numpy data loader. If set, then'
                       'train-data-path, val-data-path, and test-data-path'
                       'should also be provided.')
    group.add_argument('--num-workers', type=int, default=2,
                       help="""Number of workers to use for dataloading""")
    group.add_argument('--tokenizer-model-type', type=str,
                       default='bert-large-uncased',
                       help="Model type to use for sentencepiece tokenization \
                       (one of ['bpe', 'char', 'unigram', 'word']) or \
                       bert vocab to use for BertWordPieceTokenizer (one of \
                       ['bert-large-uncased', 'bert-large-cased', etc.])")
    group.add_argument('--tokenizer-path', type=str, default='tokenizer.model',
                       help='path used to save/load sentencepiece tokenization '
                       'models')
    group.add_argument('--tokenizer-type', type=str,
                       default='BertWordPieceTokenizer',
                       choices=['CharacterLevelTokenizer',
                                'SentencePieceTokenizer',
                                'BertWordPieceTokenizer',
                                'GPT2BPETokenizer'],
                       help='what type of tokenizer to use')
    group.add_argument("--cache-dir", default=None, type=str,
                       help="Where to store pre-trained BERT downloads")
    group.add_argument('--use-tfrecords', action='store_true',
                       help='load `--train-data`, `--valid-data`, '
                       '`--test-data` from BERT tf records instead of '
                       'normal data pipeline')
    group.add_argument('--seq-length', type=int, default=512,
                       help="Maximum sequence length to process")
    group.add_argument('--max-preds-per-seq', type=int, default=None,
                       help='Maximum number of predictions to use per sequence.'
                       'Defaults to math.ceil(`--seq-length`*.15/10)*10.'
                       'MUST BE SPECIFIED IF `--use-tfrecords` is True.')

    return parser

def get_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='PyTorch BERT Model')
    parser = add_model_config_args(parser)
    parser = add_fp16_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_text_generate_args(parser)
    parser = add_data_args(parser)

    args = parser.parse_args()

    if not args.data_dir:
        print('WARNING: No data specified')

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    if os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'):
        # We are using (OpenMPI) mpirun for launching distributed data parallel processes
        local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        local_size = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))

        # Possibly running with Slurm
        num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES', '1'))
        nodeid = int(os.getenv('SLURM_NODEID', '0'))

        args.local_rank = local_rank
        args.rank = nodeid*local_size + local_rank
        args.world_size = num_nodes*local_size

        # recover world_size and rank with Open MPI ENV
        args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
        args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))

    args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print('using world size: {} and model-parallel size: {} '.format(
            args.world_size, args.model_parallel_size))

    args.dynamic_loss_scale = False
    if args.loss_scale is None:
        args.dynamic_loss_scale = True
        if args.rank == 0:
            print(' > using dynamic loss scaling')

    # The args fp32_* or fp16_* meant to be active when the
    # args fp16 is set. So the default behaviour should all
    # be false.
    if not args.fp16:
        args.fp32_embedding = False
        args.fp32_tokentypes = False
        args.fp32_layernorm = False

    return args
