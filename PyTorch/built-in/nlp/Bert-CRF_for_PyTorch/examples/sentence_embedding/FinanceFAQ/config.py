# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# 模型文件地址
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

data_dir = 'F:/Projects/data/corpus/qa/FinanceFAQ'
q_std_file = f'{data_dir}/q_std_file.tsv'  # 标准问数据
q_corpus_file = f'{data_dir}/q_corpus_file.tsv'  # 所有语料数据
q_sim_file = f'{data_dir}/q_sim_file.tsv'

# 一阶段训练
fst_train_file = f'{data_dir}/fst_train.tsv'
fst_dev_file = f'{data_dir}/fst_dev.tsv'
ir_path = f'{data_dir}/fst_ir_corpus.tsv'
fst_q_std_vectors_file = f'{data_dir}/fst_q_std_vectors_file.npy'
fst_q_corpus_vectors_file = f'{data_dir}/fst_q_corpus_vectors_file.npy'
fst_std_data_results = f'{data_dir}/fst_std_data_results.tsv'
fst_eval_path_list = [f'{data_dir}/fst_eval.tsv']

# 二阶段
sec_train_file =  f'{data_dir}/sec_train_file.tsv'
sec_dev_file = f'{data_dir}/sec_dev_file.tsv'
sec_test_file = f'{data_dir}/sec_test_file.tsv'
sec_q_std_vectors_file = f'{data_dir}/sec_q_std_vectors_file.npy'
sec_q_corpus_vectors_file = f'{data_dir}/sec_q_corpus_vectors_file.npy'
sec_eval_path_list = []