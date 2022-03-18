import os
import sys
import numpy as np
from PIL import Image


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def deepmar_onnx(file_path, bin_path, image_info):
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    i = 0
    in_files = open(image_info, 'r').read().split('\n')[:-1]
    input_size = (224, 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for file in in_files:
        i = i + 1
        print(file, "====", i)
        img = Image.open(os.path.join(file_path, file)).convert('RGB')
        img = resize(img, input_size)

        img = np.array(img, dtype=np.float32)
        img = img / 255.

        # 均值方差
        img[..., 0] -= mean[0]
        img[..., 1] -= mean[1]
        img[..., 2] -= mean[2]
        img[..., 0] /= std[0]
        img[..., 1] /= std[1]
        img[..., 2] /= std[2]

        img = img.transpose(2, 0, 1) # HWC -> CHW
        img.tofile(os.path.join(bin_path, file.split('.')[0] + '.bin'))


if __name__ == "__main__":
    file_path = os.path.abspath(sys.argv[1])
    bin_path = os.path.abspath(sys.argv[2])
    image_info = sys.argv[3]
    deepmar_onnx(file_path, bin_path, image_info)
