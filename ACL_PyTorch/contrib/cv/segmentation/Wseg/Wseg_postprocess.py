# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import sys
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as tf
import pydensecrf.densecrf as dcrf
import torch.nn.functional as Func
from PIL import Image, ImagePalette
from pydensecrf.utils import unary_from_softmax

class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def undo(self, imgarr):
        proc_img = imgarr.copy()

        proc_img[..., 0] = (self.std[0] * imgarr[..., 0] + self.mean[0]) * 255.
        proc_img[..., 1] = (self.std[1] * imgarr[..., 1] + self.mean[1]) * 255.
        proc_img[..., 2] = (self.std[2] * imgarr[..., 2] + self.mean[2]) * 255.

        return proc_img

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

def colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'uint8'
    cmap = []
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap.append((r, g, b))
    return cmap

def get_palette():
    cmap = colormap()
    palette = ImagePalette.ImagePalette()
    for rgb in cmap:
        palette.getcolor(rgb)
    return palette

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def _cut(x_chw, pads):
    pad_h, pad_w, h, w = [int(p) for p in pads]
    return x_chw[:, pad_h:(pad_h + h), pad_w:(pad_w + w)]

def _mask_overlay(mask, image, alpha=0.3):

    mask_rgb = __mask2rgb(mask)
    return alpha * image + (1 - alpha) * mask_rgb

def __mask2rgb(mask):
    im = Image.fromarray(mask).convert("P")
    im.putpalette(get_palette())
    mask_rgb = np.array(im.convert("RGB"), dtype=np.float)
    return mask_rgb / 255.

def _merge_masks(masks, labels, pads, imsize_hw):

    mask_list = []
    for i, mask in enumerate(masks.split(1, dim=0)):

        # removing the padding
        mask_cut = _cut(mask[0], pads[i]).unsqueeze(0)
        # normalising the scale
        mask_cut = Func.interpolate(mask_cut, imsize_hw, mode='bilinear', align_corners=False)[0]

        # flipping if necessary
        if i % 2 == 1:
            mask_cut = torch.flip(mask_cut, (-1, ))

        # getting the max response
        mask_cut[1:, ::] *= labels[:, None, None]
        mask_list.append(mask_cut)

    mean_mask = sum(mask_list).numpy() / len(mask_list)

    # discounting BG
    mean_mask[0, ::] = np.power(mean_mask[0, ::], 3)

    return mean_mask

def save(out_path, img_path, img_orig, all_masks, labels, pads, gt_mask):

    img_name = os.path.basename(img_path).rstrip(".jpg")

    # converting original image to [0, 255]
    img_orig255 = np.round(255. * img_orig).astype(np.uint8)
    img_orig255 = np.transpose(img_orig255, [1, 2, 0])
    img_orig255 = np.ascontiguousarray(img_orig255)

    merged_mask = _merge_masks(all_masks, pads, labels, img_orig255.shape[:2])
    pred = np.argmax(merged_mask, 0)

    # CRF
    pred_crf = crf_inference(img_orig255, merged_mask, t=10, scale_factor=1, labels=21)
    pred_crf = np.argmax(pred_crf, 0)

    filepath = os.path.join(out_path, img_name + '.png')
    img_pred = Image.fromarray(pred.astype(np.uint8))
    img_pred.save(filepath)

    filepath = os.path.join(out_path, "crf", img_name + '.png')
    img_pred_crf = Image.fromarray(pred_crf.astype(np.uint8))
    img_pred_crf.save(filepath)
    mask_gt = gt_mask
    masks_all = np.concatenate([pred, pred_crf, mask_gt], 1).astype(np.uint8)
    images = np.concatenate([img_orig] * 3, 2)
    images = np.transpose(images, [1, 2, 0])

    overlay = _mask_overlay(masks_all, images)
    filepath = os.path.join(out_path, "vis", img_name + '.png')
    overlay255 = np.round(overlay * 255.).astype(np.uint8)
    overlay255_crf = Image.fromarray(overlay255)
    overlay255_crf.save(filepath)

def load_img_name_list(dataset_path, index=0):
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[index].strip('/') for img_gt_name in img_gt_name_list]

    return img_name_list

def load_label_name_list(dataset_path):
    return load_img_name_list(dataset_path, index=1)

def pad(image,pad_size):
    w, h = image.size

    pad_height = pad_size[0] - h
    pad_width = pad_size[1] - w

    assert pad_height >= 0 and pad_width >= 0

    pad_l = max(0, pad_width // 2)
    pad_t = max(0, pad_height // 2)

    return [pad_t, pad_l]

def imgread(imgpath):
    fullpath = os.path.join(imgpath)
    img = Image.open(fullpath).convert("RGB")
    return fullpath, img

def getitem(img_path,labelpath):

    name, img = imgread(img_path)

    assert len(labelpath) < 256, "Expected label path less than 256 for padding"

    mask = Image.open(labelpath)
    mask = np.array(mask)
    NUM_CLASS = 21
    labels = torch.zeros(NUM_CLASS - 1)

    # it will also be sorted
    unique_labels = np.unique(mask)

    # ambigious
    if unique_labels[-1] == CLASS_IDX['ambiguous']:
        unique_labels = unique_labels[:-1]

    # background
    if unique_labels[0] == CLASS_IDX['background']:
        unique_labels = unique_labels[1:]

    assert unique_labels.size > 0, 'No labels found '
    unique_labels -= 1  # shifting since no BG class
    labels[unique_labels.tolist()] = 1

    return name, img, labels, mask.astype(np.int)

def get_one_image(img_path,label_path):

    transform = tf.Compose([np.asarray,
                            Normalize()])
    pad_size = [1024, 1024]
    scales = [1, 0.5, 1.5, 2.0]
    batch_size = 8
    use_flips = True

    pad_batch = []

    for i in range(batch_size):

        sub_idx = i % batch_size
        scale = scales[sub_idx // (2 if use_flips else 1)]
        flip = use_flips and sub_idx % 2

        name, img, label, mask = getitem(img_path, label_path)

        target_size = (int(round(img.size[0] * scale)),
                       int(round(img.size[1] * scale)))

        s_img = img.resize(target_size, resample=Image.CUBIC)

        if flip:
            s_img = F.hflip(s_img)

        w, h = s_img.size
        pads_tl = pad(s_img,pad_size)
        pad_t, pad_l = pads_tl
        img = F.to_tensor(transform(img))
        pads = torch.Tensor([pad_t, pad_l, h, w])
        pad_batch.append(pads)

    return name, img, pad_batch, label, mask

def check_dir(base_path, name):

    # create the directory
    fullpath = os.path.join(base_path, name)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)

    return fullpath

def bintonp(name,bin_path):
    mask = []
    cls = []
    for i in range(8):
        msk_name = bin_path + '/' + str(name) + '_' + str(i) + '_1.bin'
        cls_name = bin_path + '/' + str(name) + '_' + str(i) + '_0.bin'
        mask_i = np.fromfile(msk_name, dtype=np.float32)
        mask_i.shape = 21,1024,1024
        cls_i = np.fromfile(cls_name, dtype=np.float32)
        cls_i.shape = 20
        cls.append(cls_i)
        mask.append(mask_i)
    msk = np.array(mask)
    clss = np.array(cls)

    return clss, msk

def denorm(image):

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    if image.dim() == 3:
        assert image.dim() == 3, "Expected image [CxHxW]"
        assert image.size(0) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip(image, MEAN, STD):
            t.mul_(s).add_(m)
    elif image.dim() == 4:
        # batch mode
        assert image.size(1) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip((0,1,2), MEAN, STD):
            image[:, t, :, :].mul_(s).add_(m)

    return image

def postprocess(file_path, voc12_root,out_path, bin_path):

    img_name_list = load_img_name_list(file_path)
    label_name_list = load_label_name_list(file_path)

    print("Start postprocess!")
    print("total image number: ",len(img_name_list))

    for i in range(len(img_name_list)):

        imgnm = img_name_list[i][33:-4]
        print("==========> ", i, "    ", imgnm)
        img_path = voc12_root + '/' + img_name_list[i]
        label_path = voc12_root + '/' + label_name_list[i]
        print(img_path)
        name, img, pad_batch, labels, gt_mask = get_one_image(img_path,label_path)

        with torch.no_grad():
            cls_raw, masks_pred = bintonp(imgnm,bin_path)
            masks_pred = torch.from_numpy(masks_pred)
            cls_raw = torch.from_numpy(cls_raw)

            cls_sigmoid = torch.sigmoid(cls_raw)
            cls_sigmoid, _ = cls_sigmoid.max(0)
            labels = (cls_sigmoid > 0.1)

        # saving the raw npy
        image = denorm(img).numpy()
        masks_pred = masks_pred.cpu()
        labels = labels.type_as(masks_pred)

        save(out_path, name, image, masks_pred, pad_batch, labels, gt_mask)


if __name__ == '__main__':

    CLASS_IDX = {
        'background': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'potted-plant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tv/monitor': 20,
        'ambiguous': 255
    }

    voc12_root_path = os.path.abspath(sys.argv[1])
    file_path = os.path.abspath(sys.argv[2])
    bin_path = os.path.abspath(sys.argv[3])    
    out_path = os.path.abspath(sys.argv[4])


    check_dir(out_path, "vis")
    check_dir(out_path, "crf")

    postprocess(file_path,voc12_root_path,out_path,bin_path)