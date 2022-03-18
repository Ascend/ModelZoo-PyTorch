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
#import os
import tempfile
import torch
import torchvision.utils as utils
import unittest


class Tester(unittest.TestCase):

    def test_make_grid_not_inplace(self):
        t = torch.rand(5, 3, 10, 10)
        t_clone = t.clone()

        utils.make_grid(t, normalize=False)
        assert torch.equal(t, t_clone), 'make_grid modified tensor in-place'

        utils.make_grid(t, normalize=True, scale_each=False)
        assert torch.equal(t, t_clone), 'make_grid modified tensor in-place'

        utils.make_grid(t, normalize=True, scale_each=True)
        assert torch.equal(t, t_clone), 'make_grid modified tensor in-place'

    def test_normalize_in_make_grid(self):
        t = torch.rand(5, 3, 10, 10) * 255
        norm_max = torch.tensor(1.0)
        norm_min = torch.tensor(0.0)

        grid = utils.make_grid(t, normalize=True)
        grid_max = torch.max(grid)
        grid_min = torch.min(grid)

        # Rounding the result to one decimal for comparison
        n_digits = 1
        rounded_grid_max = torch.round(grid_max * 10 ** n_digits) / (10 ** n_digits)
        rounded_grid_min = torch.round(grid_min * 10 ** n_digits) / (10 ** n_digits)

        assert torch.equal(norm_max, rounded_grid_max), 'Normalized max is not equal to 1'
        assert torch.equal(norm_min, rounded_grid_min), 'Normalized min is not equal to 0'

    def test_save_image(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            t = torch.rand(2, 3, 64, 64)
            utils.save_image(t, f.name)
            assert os.path.exists(f.name), 'The image is not present after save'

    def test_save_image_single_pixel(self):
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            t = torch.rand(1, 3, 1, 1)
            utils.save_image(t, f.name)
            assert os.path.exists(f.name), 'The pixel image is not present after save'


if __name__ == '__main__':
    unittest.main()
