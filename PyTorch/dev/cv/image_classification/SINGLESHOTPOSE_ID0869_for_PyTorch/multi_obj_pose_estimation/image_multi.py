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

def get_add_objs(objname):
    # Decide how many additional objects you will augment and what will be the other types of objects
    if objname == 'ape':
        add_objs = ['can', 'cat', 'duck', 'glue', 'holepuncher', 'iron', 'phone'] # eggbox
    elif objname == 'benchvise':
        add_objs = ['ape', 'can', 'cat', 'driller', 'duck', 'glue', 'holepuncher']
    elif objname == 'cam':
        add_objs = ['ape', 'benchvise', 'can', 'cat', 'driller', 'duck', 'holepuncher']
    elif objname == 'can':
        add_objs = ['ape', 'benchvise', 'cat', 'driller', 'duck', 'eggbox', 'holepuncher']
    elif objname == 'cat':
        add_objs = ['ape', 'can', 'duck', 'glue', 'holepuncher', 'eggbox', 'phone']
    elif objname == 'driller':
        add_objs = ['ape', 'benchvise', 'can', 'cat', 'duck', 'glue', 'holepuncher']
    elif objname == 'duck':
        add_objs = ['ape', 'can', 'cat', 'eggbox', 'glue', 'holepuncher', 'phone']
    elif objname == 'eggbox':
        add_objs = ['ape', 'benchvise', 'cam', 'can', 'cat', 'duck', 'glue', 'holepuncher']
    elif objname == 'glue':
        add_objs = ['ape', 'benchvise', 'cam', 'driller', 'duck', 'eggbox', 'holepuncher' ]
    elif objname == 'holepuncher':
        add_objs = ['benchvise', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox']
    elif objname == 'iron':
        add_objs = ['ape', 'benchvise', 'can', 'cat', 'driller', 'duck', 'glue']
    elif objname == 'lamp':
        add_objs = ['ape', 'benchvise', 'can', 'driller', 'eggbox', 'holepuncher', 'iron']
    elif objname == 'phone':
        add_objs = ['ape', 'benchvise', 'cam', 'can', 'driller', 'duck', 'holepuncher']
    return add_objs

def mask_background(img, mask):
    ow, oh = img.size
    
    imcs = list(img.split())
    maskcs = list(mask.split())
    fics = list(Image.new(img.mode, img.size).split())
    
    for c in range(len(imcs)):
        negmask = maskcs[c].point(lambda i: 1 - i / 255)
        posmask = maskcs[c].point(lambda i: i / 255)
        fics[c] = ImageMath.eval("a * c", a=imcs[c], c=posmask).convert('L')
    out = Image.merge(img.mode, tuple(fics))
    return out


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
    oh = img.height  
    ow = img.width
    
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

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy, num_keypoints, max_num_gt):
    
    num_labels = 2*num_keypoints+3 # +2 for width, height, +1 for class label
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

            min_x = min(xs);
            max_x = max(xs);
            min_y = min(ys);
            max_y = max(ys);
            bs[i][2*num_keypoints+1] = max_x - min_x;
            bs[i][2*num_keypoints+2] = max_y - min_y;
            
            label[cc] = bs[i]
            cc += 1
            if cc >= max_num_gt:
                break

    label = np.reshape(label, (-1))
    return label

def change_background(img, mask, bg):
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

def shifted_data_augmentation_with_mask(img, mask, shape, jitter, hue, saturation, exposure):
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
    mask_cropped = mask.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))
    
    cw, ch = cropped.size
    shift_x = random.randint(-80, 80)
    shift_y = random.randint(-80, 80)
    dx = (float(pleft)/ow)/sx - (float(shift_x)/shape[0]) # FIX HERE
    dy = (float(ptop) /oh)/sy - (float(shift_y)/shape[1]) # FIX HERE

    # dx = (float(pleft)/ow)/sx - (float(shift_x)/ow) 
    # dy = (float(ptop) /oh)/sy - (float(shift_y)/oh) 

    sized = cropped.resize(shape)
    mask_sized = mask_cropped.resize(shape)

    sized = ImageChops.offset(sized, shift_x, shift_y)
    mask_sized = ImageChops.offset(mask_sized, shift_x, shift_y)
    
    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
        mask_sized = mask_sized.transpose(Image.FLIP_LEFT_RIGHT)
        
    img = sized
    mask = mask_sized
    
    return img, mask, flip, dx,dy,sx,sy

def data_augmentation_with_mask(img, mask, shape, jitter, hue, saturation, exposure):
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
    mask_cropped = mask.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)
    mask_sized = mask_cropped.resize(shape)

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
        mask_sized = mask_sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = sized
    mask = mask_sized
    
    return img, mask, flip, dx,dy,sx,sy 

def superimpose_masked_imgs(masked_img, mask, total_mask):
    ow, oh = masked_img.size
    total_mask = total_mask.resize((ow, oh)).convert('RGB')
    
    imcs = list(masked_img.split())
    bgcs = list(total_mask.split())
    maskcs = list(mask.split())
    fics = list(Image.new(masked_img.mode, masked_img.size).split())
    
    for c in range(len(imcs)):
        negmask = maskcs[c].point(lambda i: 1 - i / 255)
        posmask = maskcs[c].point(lambda i: i / 255)
        fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
    out = Image.merge(masked_img.mode, tuple(fics))

    return out

def superimpose_masks(mask, total_mask):
    # bg: total_mask
    ow, oh = mask.size
    total_mask = total_mask.resize((ow, oh)).convert('RGB')
    
    total_maskcs = list(total_mask.split())
    maskcs = list(mask.split())
    fics = list(Image.new(mask.mode, mask.size).split())
    
    for c in range(len(maskcs)):
        negmask = maskcs[c].point(lambda i: 1 - i / 255)
        posmask = maskcs[c].point(lambda i: i)
        fics[c] = ImageMath.eval("c + b * d", b=total_maskcs[c], c=posmask, d=negmask).convert('L')
    out = Image.merge(mask.mode, tuple(fics))

    return out

def augment_objects(imgpath, objname, add_objs, shape, jitter, hue, saturation, exposure, num_keypoints, max_num_gt):

    pixelThreshold = 200
    num_labels = 2*num_keypoints+3
    random.shuffle(add_objs)
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
    maskpath = imgpath.replace('JPEGImages', 'mask').replace('/00', '/').replace('.jpg', '.png')

    # Read the image and the mask
    img = Image.open(imgpath).convert('RGB')
    iw, ih = img.size
    mask = Image.open(maskpath).convert('RGB')
    img,mask,flip,dx,dy,sx,sy = shifted_data_augmentation_with_mask(img, mask, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath, iw, ih, flip, dx, dy, 1./sx, 1./sy, num_keypoints, max_num_gt)
    total_label = np.reshape(label, (-1, num_labels))  

    # Mask the background
    masked_img = mask_background(img, mask)
    mask = mask.resize(shape)
    masked_img = masked_img.resize(shape)
    
    # Initialize the total mask and total masked image
    total_mask = mask
    total_masked_img = masked_img
    count = 1
    for obj in add_objs:
        successful = False
        while not successful:

            objpath = '../LINEMOD/' + obj + '/train.txt'
            with open(objpath, 'r') as objfile:
                objlines = objfile.readlines()
            rand_index = random.randint(0, len(objlines) - 1)
            obj_rand_img_path = '../' + objlines[rand_index].rstrip()
            obj_rand_mask_path = obj_rand_img_path.replace('JPEGImages', 'mask').replace('/00', '/').replace('.jpg', '.png')
            obj_rand_lab_path = obj_rand_img_path.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

            obj_rand_img = Image.open(obj_rand_img_path).convert('RGB')
            obj_rand_mask = Image.open(obj_rand_mask_path).convert('RGB')
            obj_rand_masked_img = mask_background(obj_rand_img, obj_rand_mask)

            obj_rand_masked_img,obj_rand_mask,flip,dx,dy,sx,sy = data_augmentation_with_mask(obj_rand_masked_img, obj_rand_mask, shape, jitter, hue, saturation, exposure)
            obj_rand_label = fill_truth_detection(obj_rand_lab_path, iw, ih, flip, dx, dy, 1./sx, 1./sy, num_keypoints, max_num_gt)
            
            # compute intersection (ratio of the object part intersecting with other object parts over the area of the object)
            xx = np.array(obj_rand_mask)
            xx = np.where(xx > pixelThreshold, 1, 0)
            yy = np.array(total_mask)
            yy = np.where(yy > pixelThreshold, 1, 0)
            intersection = (xx * yy) 
            if (np.sum(xx) < 0.01) and (np.sum(xx) > -0.01):
                successful = False
                continue
            intersection_ratio = float(np.sum(intersection)) / float(np.sum(xx))
            if intersection_ratio < 0.2:
                successful = True
                total_mask = superimpose_masks(obj_rand_mask, total_mask) #  total_mask + obj_rand_mask
                total_masked_img = superimpose_masked_imgs(obj_rand_masked_img, obj_rand_mask, total_masked_img) # total_masked_img + obj_rand_masked_img
                obj_rand_label = np.reshape(obj_rand_label, (-1, num_labels))
                total_label[count, :] = obj_rand_label[0, :] 
                count = count + 1
            else:
                successful = False

    total_masked_img = superimpose_masked_imgs(masked_img, mask, total_masked_img)

    return total_masked_img, np.reshape(total_label, (-1)), total_mask

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure, bgpath, num_keypoints, max_num_gt):
    
    # Read the background image
    bg = Image.open(bgpath).convert('RGB')
    
    # Understand which object it is and get the neighboring objects
    dirname = os.path.dirname(os.path.dirname(imgpath)) ## dir of dir of file
    objname = os.path.basename(dirname)
    add_objs = get_add_objs(objname)
    num_labels = 2*num_keypoints+3
    
    # Add additional objects in the scene, apply data augmentation on the objects
    total_masked_img, label, total_mask = augment_objects(imgpath, objname, add_objs, shape, jitter, hue, saturation, exposure, num_keypoints, max_num_gt)
    img = change_background(total_masked_img, total_mask, bg)
    lb = np.reshape(label, (-1, num_labels))
    return img,label

