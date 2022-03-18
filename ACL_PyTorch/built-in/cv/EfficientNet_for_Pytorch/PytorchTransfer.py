import sys
import os
import torch
import cv2
from PIL import Image
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable


def resnet50_onnx(input_path: str, output_path: str):
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transformer = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = val_transformer(pilimg)
    img_tensor = torch.unsqueeze(img_tensor, dim=0).float()
    img_tensor = Variable(img_tensor, requires_grad=False)
    img_tensor.reshape(1, 3, 224, 224)
    img_numpy = img_tensor.cpu().numpy()

    img_name = input_path.split('/')[-1]
    bin_name = img_name.split('.')[0] + ".bin"
    output_fl = os.path.join(output_path, bin_name)   
    # save img_tensor as binary file for om inference input
    img_numpy.tofile(output_fl)

if __name__ == "__main__":
    input_img_dir = sys.argv[1]
    output_img_dir = sys.argv[2]
    images = os.listdir(input_img_dir)
    for image_name in images:
        if not image_name.endswith(".jpeg"):
            continue
        print("start to process image {}....".format(image_name))
        path_image = os.path.join(input_img_dir, image_name)
        resnet50_onnx(path_image, output_img_dir)