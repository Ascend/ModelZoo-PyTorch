import cv2
import torch
import numpy as np
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, scale=None, offset=None):
        for t in self.transforms:
            img, boxes, labels, scale, offset = t(img, boxes, labels, scale, offset)
        return img, boxes, labels, scale, offset


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        return image.astype(np.float32), boxes, labels, scale, offset


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels, scale, offset


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels, scale, offset


# ColorJitter
class ColorJitter(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, boxes, labels, scale=None, offset=None):
        im = image.copy()
        im, boxes, labels, scale, offset = self.rand_brightness(im, boxes, labels, scale, offset)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels, scale, offset = distort(im, boxes, labels, scale, offset)
        return im, boxes, labels, scale, offset


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels, scale, offset


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels, scale, offset


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels, scale, offset


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels, scale, offset


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels, scale, offset


# RandomCrop
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            if mode is None:
                return image, boxes, labels, scale, offset

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels, scale, offset


# RandomHFlip
class RandomHFlip(object):
    def __call__(self, image, boxes, classes, scale=None, offset=None):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes, scale, offset


# Normalize image
class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.std

        return image, boxes, labels, scale, offset


# Resize
class Resize(object):
    def __init__(self, size=640, mean=None):
        self.size = size
        self.mean = np.array([v*255 for v in mean])

    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        h0, w0, _ = image.shape

        if h0 > w0:
            # resize
            r = w0 / h0
            image = cv2.resize(image, (int(r * self.size), self.size)).astype(np.float32)
            # zero padding
            h, w, _ = image.shape
            image_ = np.ones([h, h, 3]) * self.mean
            dw = h - w
            left = dw // 2
            image_[:, left:left+w, :] = image
            offset = np.array([[ left / h, 0.,  left / h, 0.]])
            scale =  np.array([[w / h, 1., w / h, 1.]])

        elif h0 < w0:
            # resize
            r = h0 / w0
            image = cv2.resize(image, (self.size, int(r * self.size))).astype(np.float32)
            # zero padding
            h, w, _ = image.shape
            image_ = np.ones([w, w, 3]) * self.mean
            dh = w - h
            top = dh // 2
            image_[top:top+h, :, :] = image
            offset = np.array([[0., top / w, 0., top / w]])
            scale = np.array([1., h / w, 1., h / w])

        else:
            # resize
            if h0 == self.size:
                image_ = image
            else:
                image_ = cv2.resize(image, (self.size, self.size)).astype(np.float32)
            offset = np.zeros([1, 4])
            scale =  1.

        if boxes is not None:
            boxes = boxes * scale + offset
        
        return image_, boxes, labels, scale, offset


# convert ndarray image to tensor type
class ToTensor(object):
    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        # to rgb
        image = image[..., (2, 1, 0)]
        return torch.from_numpy(image).permute(2, 0, 1).float(), boxes, labels, scale, offset


# TrainTransform
class TrainTransforms(object):
    def __init__(self, size=640, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            ColorJitter(),
            RandomSampleCrop(),
            RandomHFlip(),
            ToPercentCoords(),
            Resize(self.size, self.mean),
            Normalize(self.mean, self.std),
            ToTensor()
        ])

    def __call__(self, image, boxes, labels, scale=None, offset=None):
        return self.augment(image, boxes, labels, scale, offset)


# ColorTransform
class ColorTransforms(object):
    def __init__(self, size=640, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            ColorJitter(),
            RandomHFlip(),
            ToPercentCoords(),
            Resize(self.size, self.mean),
            Normalize(self.mean, self.std),
            ToTensor()
        ])

    def __call__(self, image, boxes, labels, scale=None, offset=None):
        return self.augment(image, boxes, labels, scale, offset)


# ValTransform
class ValTransforms(object):
    def __init__(self, size=640, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.augment = Compose([
            Resize(self.size, self.mean),
            Normalize(self.mean, self.std),
            ToTensor()
        ])


    def __call__(self, image, boxes=None, labels=None, scale=None, offset=None):
        return self.augment(image, boxes, labels, scale, offset)
