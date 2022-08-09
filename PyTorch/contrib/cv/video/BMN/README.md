BMN训练

```
Generative adversarial nets can be extended to a conditional model if both the generator and discriminator are conditioned on some extra information y. y could be any kind of auxiliary information,such as class labels or data from other modalities. The author perform the conditioning by feeding y into the both the discriminator and generator as additional input layer.In the generator the prior input noise pz(z), and y are combined in joint hidden representation, and the adversarial training framework allows for considerable flexibility in how this hidden representation is composed. In the discriminator x and y are presented as inputs and to a discriminative function.
```

For more detail：https://arxiv.org/abs/1907.09702

The original gpu code:https://github.com/JJBOY/BMN-Boundary-Matching-Network

## Requirements

use pytorch, you can use pip or conda to install the requirements

```
# for pip
cd $project
pip3.7 install -r requirements.txt
Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
CANN 5.0.3
```



## 数据集准备

数据集获取方式请参考开源仓BSN:https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch

文件结构如下：


```
CGAN
|--ascend_function             /解决NPU不支持的算子替换函数的目录
|-- test                               /脚本文件夹
|   |--env.sh                          /环境配置文件
|   |--train_full_1p.sh                /单卡精度测试脚本
|   |--train_full_8p.sh                /8卡精度测试脚本
|   |--train_performance_1p.sh         /单卡性能测试脚本
|   |--train_performance_8p.sh         /8卡性能测试脚本
|-- demo.py                            /例子脚本
|-- loss_function.py                      /损失函数脚本
|-- main_8p.py                           /主函数，训练启动脚本
|-- model.py                           /模型脚本
|-- opts.py                           /参数脚本
```



## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path
```

模型保存在”./checkpoint“目录下，模型生成的文件保存在”./output/result“目录下

## BMN training result

| Acc@100    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 75.11    | 65         | 1        | 10       | O1       |
| 74.82    | 553       | 8        | 10       | O1       |

## Demo

执行以下命令，程序会自动生成输入并经过网络产生输出，将输出保存在"./output/demo/demo.csv"中
```
python3.7 demo.py --data_path=real_data_path
```

