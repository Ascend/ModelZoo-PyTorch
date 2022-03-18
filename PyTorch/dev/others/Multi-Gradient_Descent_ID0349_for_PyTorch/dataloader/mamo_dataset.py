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

MamoDataset class, used by the AE Data Handler implementation
of the data handler abstract class for loading datasets.

This class is implementation of the pytorch Dataset
class and adds the functionalities required by the AE
Data Handler. Since most of the datasets used in
recommender systems are similar, this class can also be
reused by other implementations of the MAMO Data Handler.
"""

import torch
from torch.utils.data import Dataset


class MamoDataset(Dataset):
    """The MamoDataset class, used by the AE Data Handler implementation
    of the data handler abstract class for loading datasets.

    This class is implementation of the pytorch Dataset
    class and adds the functionalities required by the AE
    Data Handler. Since most of the datasets used in
    recommender systems are similar, this class can also be
    reused by other implementations of the MAMO Data Handler.

    Attributes:
        _input_data: A nD numpy array containing the input data of this dataset.
        _output_data: A nD numpy array containing the output data of this dataset.
    """

    def __init__(self, input_data, output_data=None):
        """Inits the Mamo Dataset with data.

        As an input it accepts the dataset, either with input and output data or only input data
        and inits an Mamo Dataset object. It accepts only input to be able to handle models like
        Auto Encoders where the input data is the same as the output data. Furthermore if there
        is input and output, it checks if they are of the same length.

        Args:
            input_data: A nD numpy array containing the input data of this dataset.
            output_data: A nD numpy array containing the output data of this dataset, default=None.

        Raises:
            ValueError: It is raised if the input_data is None. Additionally, Value Error will
            be raised if the dataset has an input_data and output_data, but the length of them
            is different
        """
        if input_data is None:
            raise ValueError('The input data is None, please give a valid input data.')
        self._input_data = torch.from_numpy(input_data)
        if output_data is not None:
            if len(input_data) != len(output_data):
                raise ValueError(
                    'The length of the input data must match the length of the output data!')
            self._output_data = torch.from_numpy(output_data)
        else:
            self._output_data = None

    def __len__(self):
        """A protocol that returns the number of samples in the dataset.

        This method is used by the pytorch DataLoader.
        """
        return len(self._input_data)

    def __getitem__(self, index):
        """A protocol that maps index to data sample from the dataset.

        This method is used by the pytorch DataLoader.

        Args:
            index: An integer that represents an index in the dataset.

        Returns:
            A tuple consiting of input and output data sample. If the Mamo Dataset
            is initialized with only input data it returns two times the input sample.

        Raises:
            IndexError: If the index is out of range, and IndexError is propagated.
            ValueError: If the index is not integer or slice, and ValueError is propagated.
        """
        x = self._input_data[index]
        if self._output_data is None:
            return x, x
        else:
            y = self._output_data[index]
            return x, y
