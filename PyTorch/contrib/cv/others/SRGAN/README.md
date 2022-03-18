# SRGAN

This implements training of SRGAN on  [VOC2012](http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/) dataset, mainly modified from [leftthomas/SRGAN](https://github.com/leftthomas/SRGAN). The Train dataset has 16700 images and The Val dataset has 425 images. Download the datasets from [here](https://pan.baidu.com/s/1xuFperu2WiYc5-_QXBemlA)(access code:5tzp).

## Requirements

创建新的conda环境，安装下面列出的包。（@后面为包的版本信息）

- apex @20210930
- torch @20210930
- `pip install -r requirements.txt` （若以安装高版本的对应的库，可跳过）
- CANN @5.0.3

## 数据集准备

### Train、Val Dataset

训练数据集有16700张图像，验证数据集有425张图像。可以通过这里[进行 (5tzp)](https://pan.baidu.com/s/1xuFperu2WiYc5-_QXBemlA)下载。创建一个名为 `data` 的文件夹，将下载好的数据集解压到该文件夹中。

### Test Image Dataset

测试数据集共有9张图片，其中5张为 `Set5` 中的所有图片，其余四张为仓库中的示例图片。测试结果以Set5的结果为准。数据集可以在[这里 (k9cj)](https://pan.baidu.com/s/1zTfNmjC5DOEMfC9gxOZc3g) 下载。将下载好的数据集同样解压到 `data` 文件夹中

### 最终数据集目录结构

```
data
|-- test
|   |-- SRF_2
|   |   |-- target
|   |   |-- data
|-- DIV2K_valid_HR
|-- DIV2K_train_HR
```

## Train

### 单p训练

注：若脚本不能正常运行，可以尝试使用 `dos2unix test/*` 命令转换后运行。

性能脚本

``` py
bash ./test/train_performance_1p.sh --data_path=../data
```

精度脚本

```
bash ./test/train_full_1p.sh --data_path=../data
```

### 多P训练

性能脚本：

```
bash ./test/train_performance_8p.sh --data_path=../data
```

精度脚本：

```
bash ./test/train_full_8p.sh --data_path=../data
```

使用`train_full_xx.py` 脚本时会自动运行测试脚本，训练结果和测试结果保存在 `./test/output` 路径下。

## SRGAN training result

| Device | FPS  | Epochs | AMP_Type | PSNR    | SSIM   |
| ------ | ---- | ------ | -------- | ------- | ------ |
| NPU 1p | 270  | 100    | O1       | 33.0558 | 0.9226 |
| NPU 8P | 1200 | 100    | O1       | 32.1882 | 0.9172 |
| GPU 1p | 360  | 100    | O1       | 33.4604 | 0.9308 |
| GPU 8P | 1400 | 100    | O1       | 31.0824 | 0.9191 |

### 训练结果示例 （npu_1p）

- **Set5_001.jpg**

  ![Set5_001.jpg](https://i.imgur.com/MJLS1HD.png)

- **Set5_002.jpg**

  ![Se5_002.jpg](https://i.imgur.com/AOHrLMs.png)

- **Set5_003.jpg**

  ![Set5_003.jpg](https://i.imgur.com/NRhBFHX.png)

- **Set5_004.jpg**

  ![Set5_004.jpg](https://i.imgur.com/Kh5wCGG.png)

- **Set5_005.jpg**

  ![Set5_005.jpg](https://i.imgur.com/g7oJ9VI.png)

  

