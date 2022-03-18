#!/usr/bin/python
# encoding: utf-8
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
import random
import os
from PIL import Image, ImageChops, ImageMath
import numpy as np

def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res  = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):

    ow, oh = img.size
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy, num_keypoints, max_num_gt):
    num_labels = 2 * num_keypoints + 3
    label = np.zeros((max_num_gt,num_labels))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, num_labels))
        cc = 0
        for i in range(bs.shape[0]):
            xs = list()
            ys = list()
            for j in range(num_keypoints):
                xs.append(bs[i][2*j+1])
                ys.append(bs[i][2*j+2])

            # Make sure the centroid of the object/hand is within image
            xs[0] = min(0.999, max(0, xs[0] * sx - dx)) 
            ys[0] = min(0.999, max(0, ys[0] * sy - dy)) 
            for j in range(1,num_keypoints):
                xs[j] = xs[j] * sx - dx 
                ys[j] = ys[j] * sy - dy 

            for j in range(num_keypoints):
                bs[i][2*j+1] = xs[j]
                bs[i][2*j+2] = ys[j]
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def change_background(img, mask, bg):
    # oh = img.height  
    # ow = img.width
    ow, oh = img.size
    bg = bg.resize((ow, oh)).convert('RGB')
    
    imcs = list(img.split())
    bgcs = list(bg.split())
    maskcs = list(mask.split())
    fics = list(Image.new(img.mode, img.size).split())
    
    for c in range(len(imcs)):
        negmask = maskcs[c].point(lambda i: 1 - i / 255)
        posmask = maskcs[c].point(lambda i: i / 255)
        fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
    out = Image.merge(img.mode, tuple(fics))

    return out

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure, bgpath, num_keypoints, max_num_gt):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
    maskpath = imgpath.replace('JPEGImages', 'mask').replace('/00', '/').replace('.jpg', '.png')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    mask = Image.open(maskpath).convert('RGB')
    bg = Image.open(bgpath).convert('RGB')
    
    img = change_background(img, mask, bg)
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    ow, oh = img.size
    label = fill_truth_detection(labpath, ow, oh, flip, dx, dy, 1./sx, 1./sy, num_keypoints, max_num_gt)
    return img,label

