# DeCLIP_for_PyTorch

## 1. Reference
- paper:
    - [DeCLIP: Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm](https://arxiv.org/abs/2110.05208)
- repo: 
    - https://github.com/Sense-GVT/DeCLIP
    - commit_id: 9d9e25da10e2299cf0c84b6e0be1c49085565d22 

## 2. Preparation
### 2.1 软件环境准备
    1. 安装 NPU 运行所需的driver，firmware，cann包，安装ascend—torch-1.8(当前模型仅在1.8上跑过), torch_npu, ascend-apex
    2. pip install -r requirements.txt
        提示：安装与torch版本相对应的torchvision
    3. 运行需要用到nltk的一些语料库，如果可以连接公网的话可以自动下载，如无法链接公网则需要下载以下三个链接并解压到 ~/nltk_data/corpora 
        1. https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip
        2. https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip
        3. https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip

### 2.2 数据集准备
    - 主要参考 https://github.com/Sense-GVT/DeCLIP/blob/main/docs/dataset_prepare.md#prepare-datasets
    - 需要下载准备以下几个文件，放到 ./dataset 下 ：
        1. yfcc15m_clean_open_data.json
            - 约3.3G
        2. yfcc15m_clean_open_data文件夹
            - 依据 yfcc15m_clean_open_data.json 下载得到
            - 约900G
            - 直到笔者下载时，约30w张图片无法下载，缺失或者报错
        3. bpe_simple_vocab_16e6.txt.gz
            - 约1.3M
        4. val_official.json
            - 约5.6M
        5. imagenet_val文件夹
            - imagenet valid数据集
            - 需按照ILSVRC2012_val_********.JPEG的格式放在imagenet_valid文件夹内，不包含二级目录
            - 约6.4G，5万张图片

### 2.3 原始代码仓准备
- clone了一份对应commit id的DeCLIP代码仓到```./DeCLIP```下，内部代码和github上开源代码完全一致没有修改，遵循开源代码仓相应协议和代码风格

## 3. 配置
- config.yaml来自 [link](https://github.com/Sense-GVT/DeCLIP/blob/main/experiments/declip_experiments/yfcc15m/yfcc15m_vit_declip/config.yaml)
- 数据集默认放在```./dataset```，需要修改的话，需要修改config.yaml中对应的路径
  
## 4. 执行训练脚本
- 单卡性能训练执行 `bash test/train_performance_1p.sh`
- 单卡精度训练执行 `bash test/train_full_1p.sh`
- 8卡性能训练执行 `bash test/train_performance_8p.sh`
- 8卡精度训练执行 `bash test/train_full_8p.sh`

## 5. 结果

### yfcc15m_vit_declip

#### 单机

| Eval top1 | FPS       | Npu_nums  | Steps       |
| :------:  | :------:  | :------:  | :------:    |
| -         | 85       | 1          | 100         |
| 32.91     | 680      | 8          | 128000      |



