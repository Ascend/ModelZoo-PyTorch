# StableDiffusion V2-1 for PyTorch

## 简述

StableDiffusion 是 StabilityAI公司于2022年提出的图片生成的预训练模型，论文和代码均已开源，下游任务包括文生图、图生图、图片压缩等等

- 参考实现

  ```shell
  url=https://github.com/Stability-AI/stablediffusion
  ```

- 适配昇腾AI处理器实现：

  ```shell
  url=
  ```

## 准备训练环境

### 准备环境

- 当前模型支持的pytorch版本和已知三方库依赖如下表所示。

| torch_version  | 三方库依赖版本                            |
| -------------- | ----------------------------------------- |
| PyTorch 1.11.0 | torchvision==0.12.0; transformers==4.19.2 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。

  ```
  pip install -r requirements.txt  # PyTorch1.11版本
  ```



## 模型推理



1. 下载stablediffusion v2-1预训练参数[v2-1_768-ema-pruned.ckpt](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main)到对应目录下

2. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

3. 运行文生图脚本

   ```shell
   python scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt ./checkpoints/v2-1_768-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768 --device cuda --precision full --bf16 --device_id
   4 --n_samples 1 --n_iter 1
   ```

   或者

   ```shell
   bash ./test/run_infer_full_1p.sh --ckpt_path=/data/xxx/ --n_samples=1 --device_id=0 # 单卡推理
   bash ./test/run_infer_performance_1p.sh --n_samples=1 --device_id=0  # 单卡性能
   ```

   --ckpt_path 模型文件目录，需要指定到具体的ckpt文件

   --n_samples 每次生成图片的batch数

   --n_iter 每个prompt生成的图片数，每个prompt生成的图片总数为：（n_samples * n_iter）

   --devive 除`cpu`外，默认为`cuda`

   --device_id 运行的设备ID

   

   

   torch profling

   ![image-20230509113000920](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20230509113000920.png)

   

   



