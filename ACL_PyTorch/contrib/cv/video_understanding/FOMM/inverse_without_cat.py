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


import torch
from torch.linalg import det

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cof1(mat, index):
    i0 = torch.tensor(index[0])
    i1 = torch.tensor(index[1])
    if i0.equal(torch.tensor(1)):
        ii0 = torch.tensor(1)
    else:
        ii0 = torch.tensor(0)
    if i1.equal(torch.tensor(1)):
        ii1 = torch.tensor(1)
    else:
        ii1 = torch.tensor(0)
    result = mat[ii0][ii1]
    return result


def alcof(mat, index):
    return pow(-1, index[0] + index[1]) * cof1(mat, index)


def adj(mat):
    result = torch.zeros((mat.shape[0], mat.shape[1]))
    for i in range(1, mat.shape[0] + 1):
        for j in range(1, mat.shape[1] + 1):
            result[j - 1][i - 1] = alcof(mat, [i, j])
    return result


def invmat(mat):
    cuda = torch.cuda.is_available()
    dim_0, dim_1, dim_2, dim3 = mat.shape
    if cuda:
        M_inv = torch.zeros((dim_0, dim_1, dim_2, dim3)).to(device)
    else:
        M_inv = torch.zeros((dim_0, dim_1, dim_2, dim3))
    for i in range(dim_0):
        for j in range(dim_1):
            if cuda:
                a = torch.tensor(1).to(device)
                detv = det(mat[i][j]).to(device)
                adjv = adj(mat[i][j]).to(device)
                M_inv[i][j] = a / detv * adjv
            else:
                M_inv[i][j] = 1 / det(mat[i][j]) * adj(mat[i][j])
    return M_inv