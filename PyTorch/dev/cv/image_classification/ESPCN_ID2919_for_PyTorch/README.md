# ESPCN

This repository is an attempt at implementing the paper [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158) using PyTorch.

# Setup

Please setup your training environemnt by installing the requirements using:

```
$ pip install -r requirements.txt
```

# Training

To run the model training, use the following command:

```
$ python train.py -t ./dataset/train -v ./dataset/val -o ./assets/models/
```

This will train the model and save the model's weights as `state_dict()` to the `assets/models/` folder.

Here's a GIF of model predictions during training:

![Alt Text](./assets/outputs/output.gif)

If you see the face carefully, as the model learns and the PSNR value increases, the boxes/lines from the image and face start to disappear.

# Inference

This repository comes with pre-trained model as well. To directly run inference on images, use the following command:

```
$ python infer.py -w ./assets/models/best.pth -i ./dataset/test/face.png -o ./datasets
```

**NOTE:** The pre-trained model provided with this repository is trained with a `scaling factor of 3`. Hence, the inference will up-scale the input image by a factor of 3. To use a different scaling factor, please train the model with apprropriate scaling factor value.

# Export

The `export.py` script provide five options for exporting the model - ONNX, CoreML, TensorFlow, TFLite and TF.js. For example to convert the trained model to TFLite format, use the following command:

```
$ python export.py -i ./assets/models/best.pth -o ./assets/models -f TFLite
```

# Model Statistics

The following are the model statistics for an input image of shape `[1, 1, 224, 224]`:

```
Input size (MB): 0.20
Forward/backward pass size (MB): 42.15
Params size (MB): 0.09
Estimated Total Size (MB): 42.44
------------------------
Total params: 22,729
------------------------
Total memory: 40.20MB
Total MAdd: 2.27GMAdd
Total Flops: 1.14GFlops
Total MemR+W: 38.75MB
```

# Results

The following are some images comparing the original High Resolution image vs Image up-sampled using Bi-cubic up-sampling vs Super Resolution using this model.
Note that the up-scaling factor for bi-cubic up-sampling and the super-resolution model is set to 3. For using a different up-scaling factor, please re-train the model.

<table>
    <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>ESPCN x3 (PSNR: 24.14 dB)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./dataset/test/comic.png"></center>
    	</td>
    	<td>
    		<center><img src="./assets/outputs/comic_bicubic_x3.png"></center>
    	</td>
    	<td>
    		<center><img src="./assets/outputs/comic_espcn_x3.png"></center>
    	</td>
    </tr>
  <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>ESPCN x3 (PSNR: 32.40 dB)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./dataset/test/face.png"></center>
    	</td>
    	<td>
    		<center><img src="./assets/outputs/face_bicubic_x3.png"></center>
    	</td>
    	<td>
    		<center><img src="./assets/outputs/face_espcn_x3.png"></center>
    	</td>
    </tr>
  <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>ESPCN x3 (PSNR: 28.36 dB)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./dataset/test/zebra.png"></center>
    	</td>
    	<td>
    		<center><img src="./assets/outputs/zebra_bicubic_x3.png"></center>
    	</td>
    	<td>
    		<center><img src="./assets/outputs/zebra_espcn_x3.png"></center>
    	</td>
    </tr>
  <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>ESPCN x3 (PSNR: 28.65 dB)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./dataset/test/flowers.png"></center>
    	</td>
    	<td>
    		<center><img src="./assets/outputs/flowers_bicubic_x3.png"></center>
    	</td>
    	<td>
    		<center><img src="./assets/outputs/flowers_espcn_x3.png"></center>
    	</td>
    </tr>
</table>

**NOTE:** PSNR was calculated on Y channel images.

# Contributions

If you find a bug or have any other issue/question, please create a GitHub issue, or submit a pull request.

# Credits

[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)

```
@article{DBLP:journals/corr/ShiCHTABRW16,
  author    = {Wenzhe Shi and
               Jose Caballero and
               Ferenc Husz{\'{a}}r and
               Johannes Totz and
               Andrew P. Aitken and
               Rob Bishop and
               Daniel Rueckert and
               Zehan Wang},
  title     = {Real-Time Single Image and Video Super-Resolution Using an Efficient
               Sub-Pixel Convolutional Neural Network},
  journal   = {CoRR},
  volume    = {abs/1609.05158},
  year      = {2016},
  url       = {http://arxiv.org/abs/1609.05158},
  archivePrefix = {arXiv},
  eprint    = {1609.05158},
  timestamp = {Mon, 13 Aug 2018 16:47:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/ShiCHTABRW16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

[Is the deconvolution layer the same as a convolutional layer? A note on Real­Time Single Image and Video Super­Resolution Using an Efficient Sub­Pixel Convolutional Neural Network.](https://arxiv.org/pdf/1609.07009.pdf)

```
@article{article,
author = {Shi, Wenzhe and 
          Caballero, Jose and 
          Theis, Lucas and 
          Huszar, Ferenc and 
          Aitken, Andrew and 
          Ledig, Christian and 
          Wang, Zehan},
year = {2016},
month = {09},
pages = {},
title = {Is the deconvolution layer the same as a convolutional layer?}
}
```
