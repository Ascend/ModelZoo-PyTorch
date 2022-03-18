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
"""Copyright 漏 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

This test generates random data to test the DataHandler.
"""
from dataloader.ae_data_handler import AEDataHandler
import os
import numpy as np

np.random.seed(42)
dir_path = 'test_data_aedh/'
# create temporary directory
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

train_data_path = os.path.join(
    dir_path, 'movielens_small_training.npy')
validation_input_data_path = os.path.join(
    dir_path, 'movielens_small_validation_input.npy')
validation_output_data_path = os.path.join(
    dir_path, 'movielens_small_validation_test.npy')
test_input_data_path = os.path.join(
    dir_path, 'movielens_small_test_input.npy')
test_output_data_path = os.path.join(
    dir_path, 'movielens_small_test_test.npy')

np.save(train_data_path, np.random.rand(10001, 8936).astype('float32'))
np.save(validation_input_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(validation_output_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(test_input_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(test_output_data_path, np.random.rand(2000, 8936).astype('float32'))


# Testing the length method
def test_length():
    movieLensDataHandler = AEDataHandler(
        'MovieLensSmall', train_data_path, validation_input_data_path,
        validation_output_data_path, test_input_data_path,
        test_output_data_path)
    assert 10001 == movieLensDataHandler.get_traindata_len()
    assert 2000 == movieLensDataHandler.get_testdata_len()
    assert 2000 == movieLensDataHandler.get_validationdata_len()
    assert 10001 == movieLensDataHandler.get_traindata_len()
    assert 2000 == movieLensDataHandler.get_testdata_len()
    assert 2000 == movieLensDataHandler.get_validationdata_len()


# Testing the training dataloader
def test_train_dataloader():
    movieLensDataHandler = AEDataHandler(
        'MovieLensSmall', train_data_path, validation_input_data_path,
        validation_output_data_path, test_input_data_path,
        test_output_data_path)

    # test the number of batches
    train_dataloader = movieLensDataHandler.get_train_dataloader()
    count = 0
    for batch in train_dataloader:
        assert 500 == len(batch[0])
        assert 500 == len(batch[1])
        assert 8936 == len(batch[0][0])
        assert 8936 == len(batch[1][0])
        count += 1
    assert 20 == count

    count = 0
    for batch in train_dataloader:
        assert 500 == len(batch[0])
        assert 500 == len(batch[1])
        count += 1
    assert 20 == count


# Testing the validating dataloader
def test_validation_dataloader():
    movieLensDataHandler = AEDataHandler(
        'MovieLensSmall', train_data_path, validation_input_data_path,
        validation_output_data_path, test_input_data_path,
        test_output_data_path)

    # test the number of batches
    validation_dataloader = movieLensDataHandler.get_validation_dataloader()
    count = 0
    for batch in validation_dataloader:
        assert 500 == len(batch[0])
        assert 500 == len(batch[1])
        assert 8936 == len(batch[0][0])
        assert 8936 == len(batch[1][0])
        count += 1
    assert 4 == count

    count = 0
    for batch in validation_dataloader:
        assert 500 == len(batch[0])
        assert 500 == len(batch[1])
        count += 1
    assert 4 == count


# Testing the testing dataloader
def test_test_dataloader():
    movieLensDataHandler = AEDataHandler(
        'MovieLensSmall', train_data_path, validation_input_data_path,
        validation_output_data_path, test_input_data_path,
        test_output_data_path)

    # test the number of batches
    test_dataloader = movieLensDataHandler.get_test_dataloader()
    count = 0
    for batch in test_dataloader:
        assert 500 == len(batch[0])
        assert 500 == len(batch[1])
        assert 8936 == len(batch[0][0])
        assert 8936 == len(batch[1][0])
        count += 1
    assert 4 == count

    count = 0
    for batch in test_dataloader:
        assert 500 == len(batch[0])
        assert 500 == len(batch[1])
        count += 1
    assert 4 == count


# Testing different options
def test_batchsize():
    movieLensDataHandler = AEDataHandler(
        'MovieLensSmall', train_data_path, validation_input_data_path,
        validation_output_data_path, test_input_data_path,
        test_output_data_path)

    # test the number of batches
    train_dataloader = movieLensDataHandler.get_train_dataloader(
        batch_size=200)
    count = 0
    for batch in train_dataloader:
        assert 200 == len(batch[0])
        assert 200 == len(batch[1])
        assert 8936 == len(batch[0][0])
        assert 8936 == len(batch[1][0])
        count += 1
    assert 50 == count


# Testing different options
def test_droplast():
    movieLensDataHandler = AEDataHandler(
        'MovieLensSmall', train_data_path, validation_input_data_path,
        validation_output_data_path, test_input_data_path,
        test_output_data_path)

    train_dataloader = movieLensDataHandler.get_train_dataloader(
        batch_size=200, drop_last=False)
    count = 0
    for batch in train_dataloader:
        assert 8936 == len(batch[0][0])
        assert 8936 == len(batch[1][0])
        count += 1
    assert 51 == count


def test_shuffle():
    movieLensDataHandler = AEDataHandler(
        'MovieLensSmall', train_data_path, validation_input_data_path,
        validation_output_data_path, test_input_data_path,
        test_output_data_path)

    # test the number of batches
    train_dataloader = movieLensDataHandler.get_train_dataloader(shuffle=False)

    first = True
    first_batch = None
    for batch in train_dataloader:
        if first:
            first_batch = batch
            first = False

    first = True
    for batch in train_dataloader:
        if first:
            comparison = batch[0] == first_batch[0]
            assert comparison.all()
            break


# Testing the dimension methods
def test_dimension_methods():
    movieLensDataHandler = AEDataHandler(
        'MovieLensSmall', train_data_path, validation_input_data_path,
        validation_output_data_path, test_input_data_path,
        test_output_data_path)

    assert 8936 == movieLensDataHandler.get_input_dim()
    assert 8936 == movieLensDataHandler.get_output_dim()


# removing generated data
def test_cleanup():
    os.remove(train_data_path)
    os.remove(validation_input_data_path)
    os.remove(validation_output_data_path)
    os.remove(test_input_data_path)
    os.remove(test_output_data_path)
    os.rmdir(dir_path)
