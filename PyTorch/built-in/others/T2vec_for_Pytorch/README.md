# T2vec for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

t2vec提出了一种基于深度学习的轨迹相似性计算方法，通过学习轨迹的表示向量来缓解轨迹数据中不一致采样率和噪声的影响。

- 参考实现：

  ```
  url=https://github.com/boathit/t2vec
  commit_id=942d2faca5c9b79d806f458524bb197e7516751
  ```

- 适配昇腾AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/others
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | |
  | PyTorch 1.8 | |
  | PyTorch 1.11 | |
  | PyTorch 2.1 | |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  需配套二进制包使用。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r requirements.txt
  ```

-  安装julia及其相关依赖。

    ```
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz # 根据实际架构获取

    tar xvfz julia-1.6.1-linux-x86_64.tar.gz

    sudo ln -s `realpath ./julia-1.6.1/bin/julia` /usr/local/bin/julia # 根据实际情况建立软链
    ```
    安装必要依赖
    ```

    julia pkg-install.jl # 确保此步在t2vec目录下
    ```

## 准备数据集

1. 获取数据集。

   用户自行下载 `Porto` 开源数据集，并参考源码仓对数据进行处理。

   把数据放置`项目根目录/data`下。

   数据集目录结构参考如下所示。

   ```
   ├─项目根目录/data
          |-- exp-trj.h5
          |-- exp-trj.t
          |-- porto-param-cell100
          |-- porto-vocab-dist-cell100.h5
          |-- porto.csv
          |-- porto.h5
          |-- train.mta
          |-- train.src
          |-- train.trg
          |-- training.log
          |-- trj.t
          |-- val.mta
          |-- val.src
          |-- val.trg
    ```
   > **说明：**
   >该数据集的训练过程脚本只作为一种参考示例。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```
     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     ```
   - 单机单卡评测

     启动单卡评测。

     ```
     julia evales.jl --data_path 数据集路径 --pth_path 模型权重文件路径  # 单卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   --pth_path参数填写模型权重文件路径，需写到权重文件的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --vocab_size                        //词汇大小
   --data                              //数据集路径
   --criterion_name                    //使用损失函数名
   --batch_size                        //训练批次大小
   --steps                             //训练迭代次数
   --local_rank                        //使用卡号
   ```

   训练完成后，权重文件保存在`data`下，日志信息保存在`test/output`并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | MR(dbsize:2k) | FPS   | AMP_Type    | Torch_Version |
| :------: | :---: | :--: | :------:  | :-----------: |
| 1p-竞品V | 2.421 | 982 |O1 | 1.8 |
| 1p-NPU   | 2.382 | 997 |O1 | 1.8 |


# 版本说明

## 变更

2023.6.1：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
