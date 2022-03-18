### Cross-Scale-Non-Local-Attention模型PyTorch离线推理指导

#### 1 环境准备

1.安装必要的依赖

`pip3.7 install -r requirements.txt` 

2.获取开源模型代码


```
git clone https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention.git -b master
cd Cross-Scale-Non-Local-Attention/
git reset af168f99afad2d9a04a3e4038522987a5a975e86 --hard
cd ../
```

3.获取权重文件 Cross-Scale-Non-Local-Attention预训练pth权重文件

`wget https://drive.google.com/drive/folders/1C--KFcfAsvLM4K0J6td4TqNeiaGsTXd_?usp=sharing`

解压后将model_x4.pt放在当前目录

4.获取Set5数据集

`wget https://cv.snu.ac.kr/research/EDSR/benchmark.tar`

解压并获取benchmark目录下的Set5数据集，放在当前目录

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

将benchmark.x86_64或benchmark.aarch64放到当前目录

#### 2 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲



```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path='pwd'/Set5
```


评测结果：
| 模型 | 官网pth精度 | 310离线推理精度  | 基准性能  | 310性能  |
|----|---------|---|---|---|
|  Cross-Scale-Non-Local-Attention  |    [psnr:32.68](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention)     | psnr:32.57  | 0.22fps  | 0.219fps  |



	
备注：由于内存限制，离线模型不支持多batch
