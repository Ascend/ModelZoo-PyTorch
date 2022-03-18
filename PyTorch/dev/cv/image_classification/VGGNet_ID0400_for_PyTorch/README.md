# VGGNet-PyTorch

### Update (Feb 14, 2020)

The update is for ease of use and deployment.

 * [Example: Export to ONNX](#example-export-to-onnx)
 * [Example: Extract features](#example-feature-extraction)
 * [Example: Visual](#example-visual)

It is also now incredibly simple to load a pretrained model with a new number of classes for transfer learning:

```python
from vgg_pytorch import VGG 
model = VGG.from_pretrained('vgg11', num_classes=10)
```

### Update (January 15, 2020)

This update allows you to use NVIDIA's Apex tool for accelerated training. By default choice `hybrid training precision` + `dynamic loss amplified` version, if you need to learn more and details about `apex` tools, please visit https://github.com/NVIDIA/apex.

### Update (January 9, 2020)

This update adds a visual interface for testing, which is developed by pyqt5. At present, it has realized basic functions, and other functions will be gradually improved in the future.

### Update (January 6, 2020)

This update adds a modular neural network, making it more flexible in use. It can be deployed to many common dataset classification tasks. Of course, it can also be used in your products.

### Overview
This repository contains an op-for-op PyTorch reimplementation of [VGGNet](https://arxiv.org/pdf/1409.1556.pdf).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained VGGNet models 
 * Use VGGNet models for classification or feature extraction 

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an VGGNet on your own dataset
 * Export VGGNet models for production

### Table of contents
1. [About VGG](#about-vgg)
2. [Installation](#installation)
3. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
    * [Example: Extract features](#example-feature-extraction)
    * [Example: Export to ONNX](#example-export-to-onnx)
    * [Example: Visual](#example-visual)
4. [Contributing](#contributing) 

### About VGG

If you're new to VGGNets, here is an explanation straight from the official PyTorch implementation: 

In this work we investigate the effect of the convolutional network depth on its
accuracy in the large-scale image recognition setting. Our main contribution is
a thorough evaluation of networks of increasing depth using an architecture with
very small (3 × 3) convolution filters, which shows that a significant improvement
on the prior-art configurations can be achieved by pushing the depth to 16–19
weight layers. These findings were the basis of our ImageNet Challenge 2014
submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations
generalise well to other datasets, where they achieve state-of-the-art results. We
have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.

### Installation

Install from pypi:
```bash
$ pip3 install vgg_pytorch
```

Install from source:
```bash
$ git clone https://github.com/Lornatang/VGGNet-PyTorch.git
$ cd VGGNet-PyTorch
$ pip3 install -e .
```

### Usage

#### Loading pretrained models

Load an vgg11 network:
```python
from vgg_pytorch import VGG
model = VGG.from_name("vgg11")
```

Load a pretrained vgg11: 
```python
from vgg_pytorch import VGG
model = VGG.from_pretrained("vgg11")
```

Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  vgg11          | 30.98       | 11.37       |
|  vgg11_bn       | 29.70       | 10.19       |
|  vgg13          | 30.07       | 10.75       |
|  vgg13_bn       | 28.45       | 9.63        |
|  vgg16          | 28.41       | 9.62        |
|  vgg16_bn       | 26.63       | 8.50        |
|  vgg19          | 27.62       | 9.12        |
|  vgg19_bn       | 25.76       | 8.15        |

Details about the models are below (for CIFAR10 dataset): 

|      *Name*       |*# Params*|*Top-1 Acc.*|*Pretrained?*|
|:-----------------:|:--------:|:----------:|:-----------:|
|     `vgg11`       |  132.9M  |    91.1    |      √      |
|     `vgg13`       |   133M   |    92.8    |      √      |
|     `vgg16`       |  138.4M  |    92.6    |      √      |
|     `vgg19`       |  143.7M  |    92.3    |      √      |
|-------------------|----------|------------|-------------|
|     `vgg11_bn`    |  132.9M  |    92.2    |      √      |
|     `vgg13_bn`    |   133M   |    94.2    |      √      |
|     `vgg16_bn`    |  138.4M  |    93.9    |      √      |
|     `vgg19_bn`    |  143.7M  |    93.7    |      √      |


#### Example: Classification

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`. 

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

Here's a sample execution.

```python
import json

import torch
import torchvision.transforms as transforms
from PIL import Image

from vgg_pytorch import VGG 

# Open image
input_image = Image.open("img.jpg")

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# Load class names
labels_map = json.load(open("labels_map.txt"))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify with VGG11
model = VGG.from_pretrained("vgg11")
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():
    logits = model(input_batch)
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

print("-----")
for idx in preds:
    label = labels_map[idx]
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    print(f"{label:<75} ({prob * 100:.2f}%)")
```

#### Example: Feature Extraction 

You can easily extract features with `model.extract_features`:
```python
import torch
from vgg_pytorch import VGG 
model = VGG.from_pretrained('vgg11')

# ... image preprocessing as in the classification example ...
inputs = torch.randn(1, 3, 224, 224)
print(inputs.shape) # torch.Size([1, 3, 224, 224])

features = model.extract_features(inputs)
print(features.shape) # torch.Size([1, 512, 7, 7])
```

#### Example: Export to ONNX  

Exporting to ONNX for deploying to production is now simple: 
```python
import torch 
from vgg_pytorch import VGG 

model = VGG.from_pretrained('vgg11')
dummy_input = torch.randn(16, 3, 224, 224)

torch.onnx.export(model, dummy_input, "demo.onnx", verbose=True)
```

#### Example: Visual

```text
cd $REPO$/framework
sh start.sh
```

Then open the browser and type in the browser address [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

Enjoy it.

#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

For more datasets result. Please see `research/README.md`.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit

#### Very Deep Convolutional Networks for Large-Scale Image Recognition

*Karen Simonyan, Andrew Zisserman*

##### Abstract

In this work we investigate the effect of the convolutional network depth on its accuracy in the 
large-scale image recognition setting. Our main contribution is a thorough evaluation of networks 
of increasing depth using an architecture with very small (3x3) convolution filters, which shows 
that a significant improvement on the prior-art configurations can be achieved by pushing the depth 
to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, 
where our team secured the first and the second places in the localisation and classification tracks 
respectively. We also show that our representations generalise well to other datasets, where they 
achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly 
available to facilitate further research on the use of deep visual representations in computer vision.

[paper](https://arxiv.org/abs/1409.1556)

```text
@article{VGG,
title:{Very Deep Convolutional Networks for Large-Scale Image Recognition},
author:{Karen Simonyan, Andrew Zisserman},
journal={iclr},
year={2015}
}
```