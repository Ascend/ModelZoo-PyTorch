# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

import torch
import torch.nn as nn


'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
'''


# --------------------------------------------
# SVD Orthogonal Regularization
# --------------------------------------------
def regularizer_orth(m):
    """
    # ----------------------------------------
    # SVD Orthogonal Regularization
    # ----------------------------------------
    # Applies regularization to the training by performing the
    # orthogonalization technique described in the paper
    # This function is to be called by the torch.nn.Module.apply() method,
    # which applies svd_orthogonalization() to every layer of the model.
    # usage: net.apply(regularizer_orth)
    # ----------------------------------------
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        w = m.weight.data.clone()
        c_out, c_in, f1, f2 = w.size()
        # dtype = m.weight.data.type()
        w = w.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)
        # self.netG.apply(svd_orthogonalization)
        u, s, v = torch.svd(w)
        s[s > 1.5] = s[s > 1.5] - 1e-4
        s[s < 0.5] = s[s < 0.5] + 1e-4
        w = torch.mm(torch.mm(u, torch.diag(s)), v.t())
        m.weight.data = w.view(f1, f2, c_in, c_out).permute(3, 2, 0, 1)  # .type(dtype)
    else:
        pass


# --------------------------------------------
# SVD Orthogonal Regularization
# --------------------------------------------
def regularizer_orth2(m):
    """
    # ----------------------------------------
    # Applies regularization to the training by performing the
    # orthogonalization technique described in the paper
    # This function is to be called by the torch.nn.Module.apply() method,
    # which applies svd_orthogonalization() to every layer of the model.
    # usage: net.apply(regularizer_orth2)
    # ----------------------------------------
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        w = m.weight.data.clone()
        c_out, c_in, f1, f2 = w.size()
        # dtype = m.weight.data.type()
        w = w.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)
        u, s, v = torch.svd(w)
        s_mean = s.mean()
        s[s > 1.5*s_mean] = s[s > 1.5*s_mean] - 1e-4
        s[s < 0.5*s_mean] = s[s < 0.5*s_mean] + 1e-4
        w = torch.mm(torch.mm(u, torch.diag(s)), v.t())
        m.weight.data = w.view(f1, f2, c_in, c_out).permute(3, 2, 0, 1)  # .type(dtype)
    else:
        pass



def regularizer_clip(m):
    """
    # ----------------------------------------
    # usage: net.apply(regularizer_clip)
    # ----------------------------------------
    """
    eps = 1e-4
    c_min = -1.5
    c_max = 1.5

    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        w = m.weight.data.clone()
        w[w > c_max] -= eps
        w[w < c_min] += eps
        m.weight.data = w

        if m.bias is not None:
            b = m.bias.data.clone()
            b[b > c_max] -= eps
            b[b < c_min] += eps
            m.bias.data = b

#    elif classname.find('BatchNorm2d') != -1:
#
#       rv = m.running_var.data.clone()
#       rm = m.running_mean.data.clone()
#
#        if m.affine:
#            m.weight.data
#            m.bias.data
