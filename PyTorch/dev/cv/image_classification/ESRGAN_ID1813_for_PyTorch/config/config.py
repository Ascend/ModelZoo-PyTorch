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
import argparse
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))

parser = argparse.ArgumentParser()

parser.add_argument('--image_size', type=int, default=128, help='the height / width of the hr image to network')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--sample_batch_size', type=int, default=1, help='sample batch size')
parser.add_argument('--num_epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--epoch', type=int, default=0, help='epochs in current train')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--sample_dir', default='samples', help='folder to output images and model checkpoints')

parser.add_argument('--workers', type=int, default=5, help='number of data loading workers')
parser.add_argument('--scale_factor', type=int, default=4, help='scale factor for super resolution')
parser.add_argument('--nf', type=int, default=32, help='number of filter in esrgan')
parser.add_argument('--b1', type=float, default=0.9,
                    help='coefficients used for computing running averages of gradient and its square')
parser.add_argument('--b2', type=float, default=0.999,
                    help="coefficients used for computing running averages of gradient and its square")
parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')

parser.add_argument('--p_lr', type=float, default=2e-4, help='learning rate when when training perceptual oriented')
parser.add_argument('--p_decay_iter', type=list, default=[2e5, 2 * 2e5, 3 * 2e5, 4 * 2e5, 5 * 2e5], help='batch size where learning rate halve each '
                                                                          'when training perceptual oriented')
parser.add_argument('--p_content_loss_factor', type=float, default=1, help='content loss factor when training '
                                                                           'perceptual oriented')
parser.add_argument('--p_perceptual_loss_factor', type=float, default=0, help='perceptual loss factor when training '
                                                                              'perceptual oriented')
parser.add_argument('--p_adversarial_loss_factor', type=float, default=0, help='adversarial loss factor when '
                                                                               'training perceptual oriented')

parser.add_argument('--g_lr', type=float, default=1e-4, help='learning rate when when training generator oriented')
parser.add_argument('--g_decay_iter', type=int, default=[50000, 100000, 200000, 300000], help='batch size where learning rate halve each '
                                                                          'when training generator oriented')
parser.add_argument('--g_content_loss_factor', type=float, default=1e-1, help='content loss factor when training '
                                                                              'generator oriented')
parser.add_argument('--g_perceptual_loss_factor', type=float, default=1, help='perceptual loss factor when training '
                                                                              'generator oriented')
parser.add_argument('--g_adversarial_loss_factor', type=float, default=5e-3, help='adversarial loss factor when '
                                                                                  'training generator oriented')

parser.add_argument('--is_perceptual_oriented', type=bool, default=True, help='')
# Mixed precision training parameters
parser.add_argument('--apex', action='store_true',
                    help='Use apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O2', type=str,
                    help='For apex mixed precision training'
                         'O0 for FP32 training, O1 for mixed precision training.'
                         'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')
parser.add_argument('--loss-scale-value', default=1024., type=float, help='loss scale using in amp, default -1 means dynamic')

url = ['http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
       'http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar'
       ]

parser.add_argument('--dataset_url', type=list, default=url, help='the url of DIV2K dataset for super resolution')
def get_config():
    return parser.parse_args()
