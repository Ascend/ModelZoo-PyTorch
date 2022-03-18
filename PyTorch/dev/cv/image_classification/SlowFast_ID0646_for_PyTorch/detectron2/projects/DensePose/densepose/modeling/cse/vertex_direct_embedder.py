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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle
import torch
from torch import nn

from detectron2.utils.file_io import PathManager

from .utils import normalize_embeddings


class VertexDirectEmbedder(nn.Module):
    """
    Class responsible for embedding vertices. Vertex embeddings take
    the form of a tensor of size [N, D], where
        N = number of vertices
        D = number of dimensions in the embedding space
    """

    def __init__(self, num_vertices: int, embed_dim: int):
        """
        Initialize embedder, set random embeddings

        Args:
            num_vertices (int): number of vertices to embed
            embed_dim (int): number of dimensions in the embedding space
        """
        super(VertexDirectEmbedder, self).__init__()
        self.embeddings = nn.Parameter(torch.Tensor(num_vertices, embed_dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """
        Reset embeddings to random values
        """
        self.embeddings.zero_()

    def forward(self) -> torch.Tensor:
        """
        Produce vertex embeddings, a tensor of shape [N, D] where:
            N = number of vertices
            D = number of dimensions in the embedding space

        Return:
           Full vertex embeddings, a tensor of shape [N, D]
        """
        return normalize_embeddings(self.embeddings)

    @torch.no_grad()  # pyre-ignore[56]
    def load(self, fpath: str):
        """
        Load data from a file

        Args:
            fpath (str): file path to load data from
        """
        with PathManager.open(fpath, "rb") as hFile:
            data = pickle.load(hFile)  # pyre-ignore[6]
            for name in ["embeddings"]:
                if name in data:
                    getattr(self, name).copy_(
                        torch.tensor(data[name]).float().to(device=getattr(self, name).device)
                    )
