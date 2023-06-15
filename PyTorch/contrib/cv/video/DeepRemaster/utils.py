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
import torch_npu
from torchvision import transforms
from skimage import color
import numpy as np
from PIL import Image

def convertLAB2RGB( lab ):
   lab[:, :, 0:1] = lab[:, :, 0:1] * 100   # [0, 1] -> [0, 100]
   lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] * 255 - 128, -100, 100)  # [0, 1] -> [-128, 128]
   rgb = color.lab2rgb( lab.astype(np.float64) )
   return rgb

def convertRGB2LABTensor( rgb ):
   lab = color.rgb2lab( np.asarray( rgb ) ) # RGB -> LAB L[0, 100] a[-127, 128] b[-128, 127]
   ab = np.clip(lab[:, :, 1:3] + 128, 0, 255) # AB --> [0, 255]
   ab = transforms.ToTensor()( ab ) / 255.
   L = lab[:, :, 0] * 2.55 # L --> [0, 255]
   L = Image.fromarray( np.uint8( L ) )
   L = transforms.ToTensor()( L ) # tensor [C, H, W]
   return L, ab.float()

def addMergin(img, target_w, target_h, background_color=(0,0,0)):
   width, height = img.size
   if width==target_w and height==target_h:
      return img
   scale = max(target_w,target_h)/max(width, height)
   width = int(width*scale/16.)*16
   height = int(height*scale/16.)*16
   img = transforms.Resize( (height,width), interpolation=Image.BICUBIC )( img )

   xp = (target_w-width)//2
   yp = (target_h-height)//2
   result = Image.new(img.mode, (target_w, target_h), background_color)
   result.paste(img, (xp, yp))
   return result
