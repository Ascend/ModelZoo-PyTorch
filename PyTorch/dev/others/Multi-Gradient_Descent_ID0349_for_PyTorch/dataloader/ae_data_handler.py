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

Implementation of the MAMO Data Handler for developing
AE recommender system models.

Since the MAMO framework is developed primarily for training
multi-objective AE recommender system models, this is
implementation of the MAMO Data Handler class for loading datasets
in order to training and develop such models.
"""

from dataloader.mamo_data_handler import MamoDataHandler
from dataloader.mamo_dataset import MamoDataset
from torch.utils.data import DataLoader
import numpy as np


class AEDataHandler(MamoDataHandler):
    """Implementation of the MAMO Data Handler for developing
        AE recommender system models.

    This class is implementation of the abstract class Mamo Data Handler.
    It reads the data from already preprocessed and saved numpy arrays and
    returns DataLoaders for training, validating and testing.

    Attributes:
        _train_data_path: A string contaning the path to the traning numpy array.
        _validation_input_data_path: A string contaning the path to the validating input numpy array.
        _validation_output_data_path: A string contaning the path to the validating output numpy array.
        _test_input_data_path: A string contaning the path to the testing input numpy array.
        _test_output_data_path: A string contaning the path to the testing output numpy array.
        _train_dataset: A Mamo Dataset object for the training dataset.
        _validation_dataset: A Mamo Dataset object for the validating dataset.
        _test_dataset: A Mamo Dataset object for the testing dataset.
    """

    def __init__(self, dataset_name, train_data_path, validation_input_data_path,
                 validation_output_data_path, test_input_data_path, test_output_data_path):
        """Inits a MAMO Data Handler object.

        This constructor inits a MAMO Data Handler object from preprocessed and saved numpy
        arrays. The arrays are saved in permanent storage in 'npy' format.

        Args:
        train_data_path: A string contaning the path to the traning numpy array.
        validation_input_data_path: A string contaning the path to the validating input numpy array.
        validation_output_data_path: A string contaning the path to the validating output numpy array.
        test_input_data_path: A string contaning the path to the testing input numpy array.
        test_output_data_path: A string contaning the path to the testing output numpy array.
        train_dataset: A Mamo Dataset object for the training dataset.
        validation_dataset: A Mamo Dataset object for the validating dataset.
        test_dataset: A Mamo Dataset object for the testing dataset.

        Raises:
            ValueError: It is raised if one more of the paths to the numpy arrays is None.
        """
        super().__init__(dataset_name)
        if train_data_path is None or validation_input_data_path is None or \
                validation_output_data_path is None or test_input_data_path is None or \
                test_output_data_path is None:
            raise ValueError(
                'One or more of the paths is None, please specify a valid path to numpy array.')
        self._train_data_path = train_data_path
        self._validation_input_data_path = validation_input_data_path
        self._validation_output_data_path = validation_output_data_path
        self._test_input_data_path = test_input_data_path
        self._test_output_data_path = test_output_data_path
        self._train_dataset = None
        self._validation_dataset = None
        self._test_dataset = None

    def get_train_dataloader(self, batch_size=500, shuffle=True, drop_last=True, pin_memory=True):
        """Returns a pytorch DataLoader for the training dataset.

        A DataLoader represents a Python iterable over a dataset with
        additional functions like batching, shuffling of the data, etc.
        This function creates and returns a DataLoader created on
        the training dataset.

        Args:
            batch_size: Integer, how many samples per batch to load, default=500.
            shuffle: Boolean, set to True to have the data reshuffled at every epoch, default=True.
            drop_last: Boolean, set to True to drop the last incomplete batch, default=True.

        Returns:
            Returns pytorch DataLoader object.

        Raises:
            FileNotFoundError: It is raised if the numpy data file doesn't exist.
        """
        if self._train_dataset is None:
            self._train_dataset = MamoDataset(
                np.load(self._train_data_path), None)
        return DataLoader(self._train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=True)

    def get_validation_dataloader(self, batch_size=500, shuffle=True, drop_last=False):
        """Returns a pytorch DataLoader for the validating dataset.

        A DataLoader represents a Python iterable over a dataset with
        additional functions like batching, shuffling of the data, etc.
        This function creates and returns a DataLoader created on
        the validating dataset.

        Args:
            batch_size: Integer, how many samples per batch to load, default=500.
            shuffle: Boolean, set to True to have the data reshuffled at every epoch, default=True.
            drop_last: Boolean, set to True to drop the last incomplete batch, default=True.

        Returns:
            Returns pytorch DataLoader object.

        Raises:
            FileNotFoundError: It is raised if the numpy data file doesn't exist.
        """
        if self._validation_dataset is None:
            self._validation_dataset = MamoDataset(np.load(
                self._validation_input_data_path), np.load(self._validation_output_data_path))
        return DataLoader(self._validation_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def get_test_dataloader(self, batch_size=500, shuffle=True, drop_last=True):
        """Returns a pytorch DataLoader for the testing dataset.

        A DataLoader represents a Python iterable over a dataset with
        additional functions like batching, shuffling of the data, etc.
        This function creates and returns a DataLoader created on
        the testing dataset.

        Args:
            batch_size: Integer, how many samples per batch to load, default=500.
            shuffle: Boolean, set to True to have the data reshuffled at every epoch, default=True.
            drop_last: Boolean, set to True to drop the last incomplete batch, default=True.

        Returns:
            Returns pytorch DataLoader object.

        Raises:
            FileNotFoundError: It is raised if the numpy data file doesn't exist.
        """
        if self._test_dataset is None:
            self._test_dataset = MamoDataset(
                np.load(self._test_input_data_path), np.load(self._test_output_data_path))
        return DataLoader(self._test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def get_traindata_len(self):
        """Returns the number of samples in the training dataset.

        Returns:
            Returns integer, the number of samples in the
            training dataset.
        """
        if self._train_dataset is None:
            self._train_dataset = MamoDataset(
                np.load(self._train_data_path), None)
        return self._train_dataset.__len__()

    def get_validationdata_len(self):
        """Returns the number of samples in the validating dataset.

        Returns:
            Returns integer, the number of samples in the
            validating dataset.
        """
        if self._validation_dataset is None:
            self._validation_dataset = MamoDataset(np.load(
                self._validation_input_data_path), np.load(self._validation_output_data_path))
        return self._validation_dataset.__len__()

    def get_testdata_len(self):
        """Returns the number of samples in the testing dataset.

        Returns:
            Returns integer, the number of samples in the
            testing dataset.
        """
        if self._test_dataset is None:
            self._test_dataset = MamoDataset(
                np.load(self._test_input_data_path), np.load(self._test_output_data_path))
        return self._test_dataset.__len__()

    def get_input_dim(self):
        """Returns the second dimension of the input data.

        Returns:
            Returns integer, the second dimension of the input data.
        """
        return np.load(self._test_input_data_path).shape[1]

    def get_output_dim(self):
        """Returns the second dimension of the output data.

        Returns:
            Returns integer, the second dimension of the output data.
        """
        return np.load(self._test_output_data_path).shape[1]
