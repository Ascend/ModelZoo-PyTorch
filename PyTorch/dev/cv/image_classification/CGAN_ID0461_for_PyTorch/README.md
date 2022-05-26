# DCGAN Implementation using PyTorch
DCGAN implementation for learning.

<!-- TOC -->

- [DCGAN Implementation using PyTorch](#dcgan-implementation-using-pytorch)
- [How To Run](#how-to-run)
- [Results](#results)
- [Resources Used](#resources-used)
- [Notes](#notes)
- [Other Useful Resources](#other-useful-resources)
- [TODO](#todo)

<!-- /TOC -->

# How To Run
* clone this repo using 
<pre><code>
git clone https://github.com/ashantanu/DCGAN.git
</code></pre>
* download and unzip celeba [dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to the folder name 'celeba'. I used below snippet from a udacity google colab [notebook](https://colab.research.google.com/drive/1ytjiIM_sZohV1I6p-9Cov6DtJjidJmcq)
<pre><code>
mkdir celeba && wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
</code></pre>
* control parameters in config.yml
* run main.py

# Results
Using 3 epochs on GPU

![](./generated_images.png)

GPU Training animation

![](./animation.gif)

Loss (x axis unit is 100 iterations)

![](./loss.png)


# Resources Used
* [Dataloader tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
* [Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transforms)
* [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder)
* [DCGAN Paper](https://arxiv.org/pdf/1511.06434.pdf)
* [GAN Paper](https://arxiv.org/abs/1406.2661)
* [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
* [PyTorch Layers](https://pytorch.org/docs/stable/nn.html)
* Convolutions: [Guide to Convolutions](https://arxiv.org/pdf/1603.07285.pdf) or [This Blog](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
* [Weight Initialization](https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch) and [Pytorch functions for it](https://pytorch.org/docs/stable/nn.init.html)
* [Weights in BatchNorm](https://github.com/pytorch/pytorch/issues/16149), [affine in batchnorm](https://discuss.pytorch.org/t/affine-parameter-in-batchnorm/6005/3)
* Why to use Detach in this code, and why is it not used in generator step: [1](https://github.com/pytorch/examples/issues/116) and [2](https://stackoverflow.com/questions/46944629/why-detach-needs-to-be-called-on-variable-in-this-example) 

# Notes
* For reproducibility, manually set the random of pytorch and other python libraries. Refer [this](https://pytorch.org/docs/stable/notes/randomness.html) for reproducibility pytorch using CUDA.
* GAN notes [here]()
* Transpose Convolution: Like an opposite of convolution. For ex. Maps 1x1 to 3x3. 
* Upsampling: opposite of pooling. Fills in pixels by copying pixel values, using nearest neighbor or some other method.
* For keeping track of the generator’s learning progression, generate a fixed batch of latent vectors. We can pass it through generator to see visualize how generator improves.

# Other Useful Resources
* [GAN Hacks](https://github.com/soumith/ganhacks)
* [Pytorch Autograd Tutorials](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#gradients)
* Pytorch autograd [works](https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95#:~:text=The%20leaves%20of%20this%20graph,way%20using%20the%20chain%20rule%20.)
* Google Colab: [Keep Connected](https://stackoverflow.com/questions/57113226/how-to-prevent-google-colab-from-disconnecting), [add data](https://medium.com/@prajwal.prashanth22/google-colab-drive-as-persistent-storage-for-long-training-runs-cb82bc1d5b71), [save model](https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch)
* [Why GANs are hard to train](https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b)

snippet to keep colab running:
<pre><code>
function ClickConnect(){
    console.log("Clicked on connect button"); 
    document.querySelector("colab-toolbar-button").click() // Change id here
}
setInterval(ClickConnect,60000)
</code></pre>

# TODO
* Check what is dilation in conv2d layer
* Check back-propagation in transpose convolution
* Weight initialization should use values from config
* Understand weight initialization in BatchNorm: how does it work?, what is affine used for?, how to initialize it properly
* Is there a choice to be made for sampling latent variable 
* Check why 1024 layer skipped in tutorial
