# Centermask2

This implements training of CenterMask2 on the COCO2017 dataset, mainly modified from [Centermask2](https://link.zhihu.com/?target=https%3A//github.com/youngwanLEE/CenterMask)


## Requirements

```shell
// 环境配置
//1. 使用pip安装必须依赖的包
pip install -r requirements.txt
//2. 安装axcend适配的的torch和apex
pip install torch-1.5.0+ascend.post3.20210930
pip install apex-0.1+ascend.20210930
//3. 安装torchvision==v0.5.0
git clone --branch v0.5.0 https://github.com/pytorch/vision.git
cd vision 
python setup.py build develop
pip install -e .
cd ..
//4. 安装修改过后的detectron2==v0.3
cd models/detectron2
python setup.py build develop
pip install -e .
cd ..
//5. 确认在models\centermask2\configs\centermask当中有预训练权重文件vovnet39_ese_detectron2.pth
//权重文件下载地址： https://dl.dropbox.com/s/q98pypf96rhtd8y/vovnet39_ese_detectron2.pth 
//在运行模型的过程中如果访问到模型权重文件，请将models\centermask2\configs\centermask\zsclzy_model_config_amp.yaml 文件中的MODELS\WEGHTS中的路径修改为绝对路径
```

## Training

To train a model, run `train_net.py` with the desired model architecture and the path to the COCO2017 dataset:

```bash
#training 1p performance
bash test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash test/train_performance_8p.sh --data_path=real_data_path

#test 8p accuracy
bash test/train_eval_8p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path

# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path
```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/centermask2.log  # 8p training performance result log
    test/output/devie_id/centermask2.log   # 8p training accuracy result log



## Centermask2 training result

|Bbox AP|   Segm AP   |  FPS   | Npu_nums | Iters | AMP_Type |
|:-----:| :-----: | :----: | :------: | :---: | :------: |
|-|    -    | 0.4743 |    1     |   1   |    O2    |
|13.6968| 10.4578 | 2.2928 |    8     | 3699  |    O1    |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md