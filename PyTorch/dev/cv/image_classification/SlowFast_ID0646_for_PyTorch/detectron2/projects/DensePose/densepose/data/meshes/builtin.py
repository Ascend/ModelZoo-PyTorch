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

from .catalog import MeshInfo, register_meshes

DENSEPOSE_MESHES_DIR = "https://dl.fbaipublicfiles.com/densepose/meshes/"

MESHES = [
    MeshInfo(
        name="smpl_27554",
        data="smpl_27554.pkl",
        geodists="geodists/geodists_smpl_27554.pkl",
        symmetry="symmetry/symmetry_smpl_27554.pkl",
        texcoords="texcoords/texcoords_smpl_27554.pkl",
    ),
    MeshInfo(
        name="chimp_5029",
        data="chimp_5029.pkl",
        geodists="geodists/geodists_chimp_5029.pkl",
        symmetry="symmetry/symmetry_chimp_5029.pkl",
        texcoords="texcoords/texcoords_chimp_5029.pkl",
    ),
    MeshInfo(
        name="cat_5001",
        data="cat_5001.pkl",
        geodists="geodists/geodists_cat_5001.pkl",
        symmetry="symmetry/symmetry_cat_5001.pkl",
        texcoords="texcoords/texcoords_cat_5001.pkl",
    ),
    MeshInfo(
        name="cat_7466",
        data="cat_7466.pkl",
        geodists="geodists/geodists_cat_7466.pkl",
        symmetry="symmetry/symmetry_cat_7466.pkl",
        texcoords="texcoords/texcoords_cat_7466.pkl",
    ),
    MeshInfo(
        name="sheep_5004",
        data="sheep_5004.pkl",
        geodists="geodists/geodists_sheep_5004.pkl",
        symmetry="symmetry/symmetry_sheep_5004.pkl",
        texcoords="texcoords/texcoords_sheep_5004.pkl",
    ),
    MeshInfo(
        name="zebra_5002",
        data="zebra_5002.pkl",
        geodists="geodists/geodists_zebra_5002.pkl",
        symmetry="symmetry/symmetry_zebra_5002.pkl",
        texcoords="texcoords/texcoords_zebra_5002.pkl",
    ),
    MeshInfo(
        name="horse_5004",
        data="horse_5004.pkl",
        geodists="geodists/geodists_horse_5004.pkl",
        symmetry="symmetry/symmetry_horse_5004.pkl",
        texcoords="texcoords/texcoords_zebra_5002.pkl",
    ),
    MeshInfo(
        name="giraffe_5002",
        data="giraffe_5002.pkl",
        geodists="geodists/geodists_giraffe_5002.pkl",
        symmetry="symmetry/symmetry_giraffe_5002.pkl",
        texcoords="texcoords/texcoords_giraffe_5002.pkl",
    ),
    MeshInfo(
        name="elephant_5002",
        data="elephant_5002.pkl",
        geodists="geodists/geodists_elephant_5002.pkl",
        symmetry="symmetry/symmetry_elephant_5002.pkl",
        texcoords="texcoords/texcoords_elephant_5002.pkl",
    ),
    MeshInfo(
        name="dog_5002",
        data="dog_5002.pkl",
        geodists="geodists/geodists_dog_5002.pkl",
        symmetry="symmetry/symmetry_dog_5002.pkl",
        texcoords="texcoords/texcoords_dog_5002.pkl",
    ),
    MeshInfo(
        name="dog_7466",
        data="dog_7466.pkl",
        geodists="geodists/geodists_dog_7466.pkl",
        symmetry="symmetry/symmetry_dog_7466.pkl",
        texcoords="texcoords/texcoords_dog_7466.pkl",
    ),
    MeshInfo(
        name="cow_5002",
        data="cow_5002.pkl",
        geodists="geodists/geodists_cow_5002.pkl",
        symmetry="symmetry/symmetry_cow_5002.pkl",
        texcoords="texcoords/texcoords_cow_5002.pkl",
    ),
    MeshInfo(
        name="bear_4936",
        data="bear_4936.pkl",
        geodists="geodists/geodists_bear_4936.pkl",
        symmetry="symmetry/symmetry_bear_4936.pkl",
        texcoords="texcoords/texcoords_bear_4936.pkl",
    ),
]

register_meshes(MESHES, DENSEPOSE_MESHES_DIR)
