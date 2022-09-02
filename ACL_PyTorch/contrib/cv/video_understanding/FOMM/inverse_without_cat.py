from torch import inverse
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