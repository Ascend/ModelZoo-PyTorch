# ABINet_MMOCR

- [概述](#ABSTRACT)
- [环境准备](#ENV_PREPARE)
- [准备数据集]()
- [快速上手](#QUICK_START)
- [模型推理性能&精度](#INFER_PERFORM)

***

## 概述 <a name="ABSTRACT"></a>
本模块展现的是针对openmmlab中开发的ABINet模型进行了适配昇腾pytorch插件的样例。本样例展现了如何使用mmocr将模型进行转换并通过昇腾pytorch插件将其赋予昇腾推理引擎的能力并在npu上高性能地运行。
- 模型链接
    ```
    url=https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/abinet/README.md
    ```
- 模型对应配置文件
    ```
    url=https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/abinet/abinet_20e_st-an_mj.py
    ```
- 模型权重
  ```
  url=https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_20e_st-an_mj/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth
  ```

## 环境准备 <a name="ENV_PREPARE"></a>
| 配套                   | 版本            | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 | 链接                                                          |
| Python                | 3.9.0           |                                                           
| PyTorch               | 2.0.1+cpu           |
| torchVison            | 0.15.2          |-
| Ascend-cann-torch-aie | --
| Ascend-cann-aie       | --
| 芯片类型               | Ascend310P3     |
### 配置OpenMMLab运行环境
运行基于openmmlab推理框架的abinet等模型进行推理前，需要提前根据openmmlab的在线指导文档安装部署[mmocr](https://github.com/open-mmlab/mmocr)，用于文字识别。
参考命令如下：
```
mkdir open-mmlab
cd open-mmlab
conda create -n open-mmlab python=3.9 pytorch=2.0.1 -c pytorch
conda activate open-mmlab
pip3 install openmim
mim install mmengine
mim install "mmcv>=2.0.0rc2"
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
mim install -e .

```
修改如下文件，用于适配流程
```
pip3 show mmocr
在 /rootpath/mmocr/models/textrecog/decoders/abi_language_decoder.py 中的_get_length函数
把 out = ((out.cumsum(dim) == 1) & out).max(dim)[1] 改成
out = ((out.int().cumsum(dim) == 1) & out.int()).int().max(dim)[1]

把 return out  改成
return out.int()


pip3 show mmcv
在 /rootpath/mmcv/cnn/bricks/transformer.py 中的forward函数
把 attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)] 改成
attn_masks = [attn_masks.clone() for _ in range(self.num_attn)]
```
### 配置昇腾运行环境
下载对应版本的昇腾产品
#### 安装CANN包

```
chmod +x Ascend-cann-toolkit_6.3.RC2.alpha002_linux-${arch}.run 
./Ascend-cann-toolkit_6.3.RC2.alpha002_linux-${arch}.run --install

source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
> ${arch}是服务器的架构,默认安装路径在`/usr/local/Ascend/ascend-toolkit`下

#### 安装推理引擎

```
chmod +x Ascend-cann-aie_6.3.T200_linux-${arch}.run
./Ascend-cann-aie_6.3.T200_linux-${arch}.run --install

source /usr/local/Ascend/aie/set_env.sh
```
> ${arch}是服务器的架构,默认安装路径在`/usr/local/Ascend/aie`下

#### 安装torch_aie

```
tar -zxvf Ascend-cann-torch-aie-6.3.T200-linux_${arch}.tar.gz
pip3 install torch-aie-6.3.T200-linux_${arch}.whl
```


### 准备脚本与必要文件
在本地的mmocr地址下载本代码仓中abinet_sample.py,需要注意的是，脚本里的文件路径需要与实际文件路径对齐。参考目录结构：
```
open-mmlab/
|-- abinet_sample.py
|-- mmocr
    |-- CITATION.cff
    |-- ....
    |-- setup.py
    |-- tests
    `-- tools
```

## 准备数据集 <a name="DATASET_PREPARE"></a>
clone了mmocr仓后执行下面命令可以跑原仓的样例
```
  cd mmocr
  python3 tools/infer.py demo/demo_text_recog.jpg --rec abinet --show --print-result
  cd ..
```
> 运行结束可以看到识别结果和对应评分
## 快速上手 <a name="QUICK_START"></a>

- 使用torch-aie编译模型并推理，静态场景（输入宽高一致）参考命令：

  ```
    wget https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_20e_st-an_mj/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth

    # 精度验证
    python3 abinet_sample.py 
            --pth=./abinet_20e_st-an_mj_20221005_012617-ead8c139.pth 
            --img=./mmocr/demo/demo_text_recog.jpg
            --print

    # 性能验证
    python3 abinet_sample.py 
            --pth=./abinet_20e_st-an_mj_20221005_012617-ead8c139.pth 
            --img=./mmocr/demo/demo_text_recog.jpg
            --warmup=5
            --bs=8
  ```
  > 参数详细说明可以运行 `python3 abinet_sample.py -h` 查看


## 模型推理性能&精度 <a name="INFER_PERFORM"></a>
| 芯片型号 | Batch Size | 数据集    | 性能(吞吐量) | 精度 |
|---------|------------|-----------|------|------|
| 310P3   | 1          | - | 22.174 | - |
| 310P3   | 8          | - | 57.530 | - |
