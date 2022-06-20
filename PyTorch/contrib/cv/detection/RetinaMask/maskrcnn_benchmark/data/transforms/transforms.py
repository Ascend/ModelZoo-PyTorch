# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import math
import torch

from torchvision.transforms import functional as F
from maskrcnn_benchmark.structures.segmentation_mask import Polygons


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        target.size_before_pad = (size[1], size[0])
        return image, target


class MultiScaleResize(object):
    def __init__(self, min_sizes, max_size):
        self.resizers = []
        for min_size in min_sizes:
            self.resizers.append(Resize(min_size, max_size))

    def __call__(self, image, target):
        resizer = random.choice(self.resizers)
        image, target = resizer(image, target)

        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ImgPad(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.fix_shape = cfg.INPUT.FIX_SHAPE
        self.amp = cfg.AMP
        self.opt_level = cfg.OPT_LEVEL

    def _pad(self, image, target):

        pad_value = 0

        dst_shape = (3, self.fix_shape[1], self.fix_shape[0])
        padding_size = [0, dst_shape[-1] - image.shape[-1],
                        0, dst_shape[-2] - image.shape[-2]]
        padded = torch.nn.functional.pad(image, padding_size, value=pad_value)

        if self.amp and (self.opt_level == "O1" or self.opt_level == "O2"):
            padded = padded.to(torch.float16)
        image_preprocess = padded.contiguous()

        target.size = (self.fix_shape[1], self.fix_shape[0])
        target.extra_fields['masks'].size = (self.fix_shape[1], self.fix_shape[0])
        for i in range(len(target.extra_fields['masks'].polygons)):
            target.extra_fields['masks'].polygons[i].size = (self.fix_shape[1], self.fix_shape[0])

        return image_preprocess, target

    def __call__(self, image, target):
        image, target = self._pad(image, target)

        return image, target


class BoxMaskPad(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.amp = cfg.AMP
        self.opt_level = cfg.OPT_LEVEL

    def _pad(self, target):

        boxes_num = target.bbox.shape[0]

        max_len = int(math.ceil(boxes_num / 20)) * 20

        if boxes_num < max_len:
            diff_num = max_len - boxes_num
            target.bbox = torch.cat([target.bbox, torch.zeros([diff_num, 4])], dim=0).contiguous()

            target.extra_fields['labels'] = torch.cat(
                [target.extra_fields['labels'].long(), torch.full((diff_num,), -1, dtype=torch.long)], dim=0)

            target.extra_fields['masks'].polygons += [Polygons(p, target.extra_fields['masks'].size, mode=None) for p in
                                                      [torch.zeros([1], dtype=torch.float16)] * diff_num]

        else:
            select_idx = torch.randperm(boxes_num)[:max_len]
            # noinspection PyInterpreter
            target.bbox = target.bbox[select_idx].contiguous()
            target.extra_fields['labels'] = target.extra_fields['labels'][select_idx].long().contiguous()

            target.extra_fields['masks'].polygons = [target.extra_fields['masks'].polygons[idx]
                                                     for idx in select_idx.numpy().tolist()]

        return target

    def __call__(self, image, target):

        target = self._pad(target)

        return image, target
