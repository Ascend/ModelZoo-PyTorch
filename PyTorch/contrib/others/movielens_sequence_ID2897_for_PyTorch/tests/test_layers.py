# Copyright 2018 The Cornac Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest
import torch

import torch.nn as nn

from spotlight.layers import BloomEmbedding, ScaledEmbedding


@pytest.mark.parametrize('embedding_class', [
    nn.Embedding,
    ScaledEmbedding,
    BloomEmbedding
])
def test_embeddings(embedding_class):

    num_embeddings = 1000
    embedding_dim = 16

    batch_size = 32
    sequence_length = 8

    layer = embedding_class(num_embeddings,
                            embedding_dim)

    # Test 1-d inputs (minibatch)
    indices = torch.from_numpy(
        np.random.randint(0, num_embeddings, size=batch_size, dtype=np.int64))
    representation = layer(indices)
    assert representation.size()[0] == batch_size
    assert representation.size()[-1] == embedding_dim

    # Test 2-d inputs (minibatch x sequence_length)
    indices = torch.from_numpy(
        np.random.randint(0, num_embeddings,
                          size=(batch_size, sequence_length), dtype=np.int64))
    representation = layer(indices)
    assert representation.size() == (batch_size, sequence_length, embedding_dim)
