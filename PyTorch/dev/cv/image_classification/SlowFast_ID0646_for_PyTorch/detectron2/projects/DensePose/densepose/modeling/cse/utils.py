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

import torch
from torch.nn import functional as F


def squared_euclidean_distance_matrix(pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    """
    Get squared Euclidean Distance Matrix
    Computes pairwise squared Euclidean distances between points

    Args:
        pts1: Tensor [M x D], M is the number of points, D is feature dimensionality
        pts2: Tensor [N x D], N is the number of points, D is feature dimensionality

    Return:
        Tensor [M, N]: matrix of squared Euclidean distances; at index (m, n)
            it contains || pts1[m] - pts2[n] ||^2
    """
    edm = torch.mm(-2 * pts1, pts2.t())
    edm += (pts1 * pts1).sum(1, keepdim=True) + (pts2 * pts2).sum(1, keepdim=True).t()
    return edm.contiguous()


def normalize_embeddings(embeddings: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Normalize N D-dimensional embedding vectors arranged in a tensor [N, D]

    Args:
        embeddings (tensor [N, D]): N D-dimensional embedding vectors
        epsilon (float): minimum value for a vector norm
    Return:
        Normalized embeddings (tensor [N, D]), such that L2 vector norms are all equal to 1.
    """
    return embeddings / torch.clamp(
        embeddings.norm(p=None, dim=1, keepdim=True), min=epsilon  # pyre-ignore[6]
    )


def get_closest_vertices_mask_from_ES(
    E: torch.Tensor,
    S: torch.Tensor,
    h: int,
    w: int,
    mesh_vertex_embeddings: torch.Tensor,
    device: torch.device,
):
    """
    Interpolate Embeddings and Segmentations to the size of a given bounding box,
    and compute closest vertices and the segmentation mask

    Args:
        E (tensor [1, D, H, W]): D-dimensional embedding vectors for every point of the
            default-sized box
        S (tensor [1, 2, H, W]): 2-dimensional segmentation mask for every point of the
            default-sized box
        h (int): height of the target bounding box
        w (int): width of the target bounding box
        mesh_vertex_embeddings (tensor [N, D]): vertex embeddings for a chosen mesh
            N is the number of vertices in the mesh, D is feature dimensionality
        device (torch.device): device to move the tensors to
    Return:
        Closest Vertices (tensor [h, w]), int, for every point of the resulting box
        Segmentation mask (tensor [h, w]), boolean, for every point of the resulting box
    """
    embedding_resized = F.interpolate(E, size=(h, w), mode="bilinear")[0].to(device)
    coarse_segm_resized = F.interpolate(S, size=(h, w), mode="bilinear")[0].to(device)
    mask = coarse_segm_resized.argmax(0) > 0
    closest_vertices = torch.zeros(mask.shape, dtype=torch.long, device=device)
    all_embeddings = embedding_resized[:, mask].t()
    size_chunk = 10_000  # Chunking to avoid possible OOM
    edm = []
    if len(all_embeddings) == 0:
        return closest_vertices, mask
    for chunk in range((len(all_embeddings) - 1) // size_chunk + 1):
        chunk_embeddings = all_embeddings[size_chunk * chunk : size_chunk * (chunk + 1)]
        edm.append(
            torch.argmin(
                squared_euclidean_distance_matrix(chunk_embeddings, mesh_vertex_embeddings), dim=1
            )
        )
    closest_vertices[mask] = torch.cat(edm)
    return closest_vertices, mask
