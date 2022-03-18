# WaveGlow

本项目实现了 WaveGlow 在 NPU 上的训练, 迁移自[NVIDIA/WaveGlow](https://github.com/NVIDIA/waveglow)

## WaveGlow更改信息

本项目对 NVIDIA/WaveGlow 做了如下更改：
- 迁移到 NPU 上
- 使用NpuFusedAdam优化器加速训练
- 对于一些操作，使用 NPU 算子优化性能(Conv1D替换为Conv2D)、同时将一些操作转移到 CPU 上进行(Unfold、ConvTranse1D)
- 为优化性能，将TransposeD和TransData算子的耗时较大的input shapes或output shapes加入相应文件白名单。参考链接：[TransposeD和TransData算子优化案例](https://gitee.com/wangjiangben_hw/ascend-pytorch-crowdintelligence-doc/blob/master/pytorch-train-guide/TransposeD%E5%92%8CTransData%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96%E6%A1%88%E4%BE%8B.docx)具体修改三处：
    - 优化Transpose: 在/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/opp/op_impl/built-in/ai_core/tbe/impl/dynamic/transpose.py内_by_dynamic_static_union_version函数的white_list_shape末尾添加input shapes:[8, 1, 8, 1], [6, 1, 6, 1], [4, 1, 4, 1], [24, 80, 2000, 8],[24, 2000, 8, 1], [24, 2000, 640, 1], [24, 80, 8, 2000]
    - 优化TransData: 在/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/opp/op_impl/built-in/ai_core/tbe/impl/four_2_five.py内four_2_five函数的"if src_format.upper() == "NCHW" and shape_input in"语句后面的列表中添加input shapes:[24, 8, 2000, 1], [24, 4, 2000, 1], [24, 640, 2000, 1], [24, 256, 2000, 1], [24, 6, 2000, 1], [24, 3, 2000, 1], [24, 2, 2000, 1], [24, 512, 2000, 1], [24, 4096, 2000, 1]
    - 优化TransData: 在/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/opp/op_impl/built-in/ai_core/tbe/impl/five_2_four.py内five_2_four函数的"elif dst_format.lower() == "nchw" and dst_shape in"语句后面的列表中添加output shapes:[24, 2, 2000, 1], [24, 3, 2000, 1], [24, 4, 2000, 1], [24, 6, 2000, 1], [24, 8, 2000, 1], [24, 256, 2000, 1], [24, 512, 2000, 1], [24, 640, 2000, 1], [24, 4096, 2000, 1]

## 依赖安装
本项目依赖如下：
- NPU 配套的 run 包安装(20211018_FA2.0.T308  20211105_CANN5031)
- Python 3.7.5
- apex               0.1+ascend.20210930
- appdirs            1.4.4
- audioread          2.1.9
- certifi            2018.10.15
- cffi               1.15.0
- charset-normalizer 2.0.9
- cycler             0.11.0
- decorator          5.1.0
- fonttools          4.28.5
- future             0.18.2
- idna               3.3
- inflect            5.3.0
- joblib             1.1.0
- kiwisolver         1.3.2
- librosa            0.8.1
- llvmlite           0.37.0
- matplotlib         3.5.1
- mpmath             1.2.1
- numba              0.54.1
- numpy              1.20.3
- packaging          21.3
- Pillow             8.4.0
- pip                21.3.1
- pooch              1.5.2
- pycparser          2.21
- pyparsing          3.0.6
- python-dateutil    2.8.2
- requests           2.26.0
- resampy            0.2.2
- scikit-learn       1.0.1
- scipy              1.7.3
- setuptools         40.4.3
- six                1.16.0
- SoundFile          0.10.3.post1
- sympy              1.9
- threadpoolctl      3.0.0
- torch              1.5.0+ascend.post3.20210930
- Unidecode          1.3.2
- urllib3            1.26.7
- wheel              0.32.1

完整安装方法如下。强烈推荐使用conda安装依赖，否则可能会遇到版本不对应的问题：
```bash
# 安装conda环境
conda create -n waveglow-env python=3.7.5
conda activate waveglow-env
# 安装torch和apex(这里以arm架构为例，若系统为x86架构请换成对应whl)
pip install torch-1.5.0+ascend.post3.20210930-cp37-cp37m-linux_aarch64.whl
pip install apex-0.1+ascend.20210930-cp37-cp37m-linux_aarch64.whl
# 安装依赖
pip install -r requirement.txt

# 检查安装依赖是否成功
import librosa
# 若import librosa提示 sndfile 找不到
conda install libsndfile
# 若 scikit-learn 报错，则使用conda安装
conda install scikit-learn
# 若numpy被前面安装的内容更新，则重新安装(numpy版本需1.20及以下)
pip install numpy==1.20
# 若 sympy 报错，则pip/conda安装
pip install sympy
```

## 数据集
- 本项目使用的数据集为 LJ-Speech 数据集，放在 /opt/npu/waveglow_dataset 目录下。

## 训练

```bash
cd WaveGlow
conda activate waveglow-env
source test/env_npu.sh      # 环境变量
# training 1p performance
bash test/train_performance_1p.sh --data_path=/opt/npu/waveglow_dataset \
                                  --output_directory=checkpoints

# training 8p accuracy
bash test/train_full_8p.sh --data_path=/opt/npu/waveglow_dataset \
                           --output_directory=checkpoints

# training 8p performance
bash test/train_performance_8p.sh --data_path=/opt/npu/waveglow_dataset \
                                  --output_directory=checkpoints

# finetuning 1p
bash test/train_finetune_1p.sh --data_path=/opt/npu/waveglow_dataset \
                               --output_directory=checkpoints \
                               --checkpoint_path=checkpoints/waveglow_1000
```
## 生成音频
```bash
# eval to generate audio
bash test/train_eval_8p.sh --data_path=/opt/npu/waveglow_dataset \
                           --pth_path=checkpoints/waveglow_21000 \
                           --output_directory=checkpoints
```

Log path:
- test/output/devie_id/train_device_id.log           # training detail log
- test/output/devie_id/train_WaveGlow_for_PyTorch_bs24_8p_acc_loss.txt  # 8p training accuracy result txt
- test/output/devie_id/WaveGlow_for_PyTorch_bs24_8p_acc.log   # 8p training accuracy and performance result log

## WaveGlow 训练结果

| Accuracy |    FPS    | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 0.002     | 1        | 1        | O2       |
| -5.6     | 0.21      | 8        | 313      | O2       |
