# Quadratic video interpolation

Implementation of "Quadratic video interpolation", NeurIPS 2019.

[Paper](https://arxiv.org/abs/1911.00627), [Project](https://sites.google.com/view/xiangyuxu/qvi_nips19)

## 精度指标
| Device      | Loss |
|:-----------:|:-----------:|
| GPU      | 3.309       |
| NPU   | 3.547        |

## Packages
The following pakages are required to run the code:
* python==3.8
* pytorch==1.5.1
* cudatoolkit==10.1
* torchvision==0.6.1
* cupy==8.6.0
* tensorboardX
* opencv-python
* easydict

## Video
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/vemHEbkWMAI/0.jpg)](https://www.youtube.com/watch?v=vemHEbkWMAI)


## Demo
* Download [pretrained model](https://www.dropbox.com/s/5auirpk1tijxo1v/model.pt?dl=0) and put it in "./qvi_release"
* Download [weights of PWC-Net](https://www.dropbox.com/s/afsqhlzu0rarpmf/pwc-checkpoint.pt?dl=0) and put it in "./utils"
* Put your sequence in "datasets/example" with structure as 
```
seq1
--00000.png 
--00001.png
--... 
seq2
--00000.png 
--00001.png
--... 
```

* Then run the demo:
```
python demo.py configs/test_config.py
```
The output will be in "outputs/example". Note that all settings are in config files under the folder "./configs".



## Train
* Download the [QVI-960](https://www.dropbox.com/s/4i6ff6o62sp2f69/QVI-960.zip?dl=0) dataset for training and put it in the folder "datasets"
* Download the [validation](https://www.dropbox.com/s/u50kpbj08cuucmu/Adobe240_validation.zip?dl=0) data which is a subset of the Adobe-240 dataset, and put it in the folder "datasets"
* Then run the training code:

```
python train.py configs/train_config.py
```

## Test
More datasets for evaluation:
* [Adobe-240 (full version)](https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fs%2Fpwjbbrcyk1woqxu%2Fadobe240.zip%3Fdl%3D0&sa=D&sntz=1&usg=AFQjCNHsNzXN1lu-LohDckNdFvIcJZmv4w)
* [GoPro](https://drive.google.com/file/d/1SlURvdQsokgsoyTosAaELc4zRjQz9T2U/view)
* [UCF](https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fs%2Fdbihqk5deobn0f7%2Fucf101_extracted.zip%3Fdl%3D0&sa=D&sntz=1&usg=AFQjCNE8CyLdENKhJf2eyFUWu6G2D1iJUQ)
* [Davis](https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fs%2F9t6x7fi9ui0x6bt%2Fdavis-90.zip%3Fdl%3D0&sa=D&sntz=1&usg=AFQjCNG7jT-Up65GD33d1tUftjPYNdQxkg)

You can use "datas/Sequence.py" to conveniently load the test datasets.

&nbsp;
&nbsp;

Please consider citing this paper if you find the code and data useful in your research:
```
@inproceedings{qvi_nips19,
	title={Quadratic video interpolation},
	author={Xiangyu Xu and Li Siyao and Wenxiu Sun and Qian Yin and Ming-Hsuan Yang},
	booktitle = {NeurIPS},
	year={2019}
}
```


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md