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

import json
import logging
from typing import List, Optional
import torch
from torch import nn

from detectron2.utils.file_io import PathManager

from densepose.structures.mesh import create_mesh


class MeshAlignmentEvaluator:
    """
    Class for evaluation of 3D mesh alignment based on the learned vertex embeddings
    """

    def __init__(self, embedder: nn.Module, mesh_names: Optional[List[str]]):
        self.embedder = embedder
        # use the provided mesh names if not None and not an empty list
        self.mesh_names = mesh_names if mesh_names else embedder.mesh_names
        self.logger = logging.getLogger(__name__)
        with PathManager.open(
            "https://dl.fbaipublicfiles.com/densepose/data/cse/mesh_keyvertices_v0.json", "r"
        ) as f:
            self.mesh_keyvertices = json.load(f)

    def evaluate(self):
        ge_per_mesh = {}
        gps_per_mesh = {}
        for mesh_name_1 in self.mesh_names:
            avg_errors = []
            avg_gps = []
            embeddings_1 = self.embedder(mesh_name_1)
            keyvertices_1 = self.mesh_keyvertices[mesh_name_1]
            keyvertex_names_1 = list(keyvertices_1.keys())
            keyvertex_indices_1 = [keyvertices_1[name] for name in keyvertex_names_1]
            for mesh_name_2 in self.mesh_names:
                if mesh_name_1 == mesh_name_2:
                    continue
                embeddings_2 = self.embedder(mesh_name_2)
                keyvertices_2 = self.mesh_keyvertices[mesh_name_2]
                sim_matrix_12 = embeddings_1[keyvertex_indices_1].mm(embeddings_2.T)
                vertices_2_matching_keyvertices_1 = sim_matrix_12.argmax(axis=1)
                mesh_2 = create_mesh(mesh_name_2, embeddings_2.device)
                geodists = mesh_2.geodists[
                    vertices_2_matching_keyvertices_1,
                    [keyvertices_2[name] for name in keyvertex_names_1],
                ]
                Current_Mean_Distances = 0.255
                gps = (-(geodists ** 2) / (2 * (Current_Mean_Distances ** 2))).exp()
                avg_errors.append(geodists.mean().item())
                avg_gps.append(gps.mean().item())

            ge_mean = torch.as_tensor(avg_errors).mean().item()
            gps_mean = torch.as_tensor(avg_gps).mean().item()
            ge_per_mesh[mesh_name_1] = ge_mean
            gps_per_mesh[mesh_name_1] = gps_mean
        ge_mean_global = torch.as_tensor(list(ge_per_mesh.values())).mean().item()
        gps_mean_global = torch.as_tensor(list(gps_per_mesh.values())).mean().item()
        per_mesh_metrics = {
            "GE": ge_per_mesh,
            "GPS": gps_per_mesh,
        }
        return ge_mean_global, gps_mean_global, per_mesh_metrics
