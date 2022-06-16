# U2-Net模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装
```shell
pip3.7 install -r requirements.txt
```

2.获取，修改与安装开源模型代码
```shell
mkdir workspace && cd workspace
git clone https://github.com/xuebinqin/U-2-Net.git 
cd U-2-Net
git reset a179b4bfd80f84dea2c76888c0deba9337799b60 --hard
cd ..
git clone https://gitee.com/Ronnie_zheng/MagicONNX
cd MagicONNX && git checkout af31a42e3873f2726af3336c472b7485dcf487c1
cd ..
```

3.获取权重文件

获取权重文件：u2net.pth，放置到工作目录:
```shell
mkdir -p ./workspace/U-2-Net/save_models/u2net
cp u2net.pth ./workspace/U-2-Net/save_models/u2net/
```

4.数据集

获取ECSSD，解压到当前目录:
```shell
mkdir -p datasets
cp -r ECSSD ./datasets/
```


5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64放到当前目录

## 2 离线推理

310P上执行, 执行时使npu-smi info查看设备状态，确保device空闲：

${chip_name}可通过`npu-smi info`指令查看

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
bash test/pth2om.sh Ascend${chip_name} # Ascend310P3
bash test/eval_acc_perf.sh
```

GPU机器上执行，执行时使用nvidia-smi查看设备状态，确保device空闲：

```
bash perf_g.sh
```

 **评测结果：**


| 模型        | pth精度              | 310P离线推理精度      | 基准性能    | 310P性能  |
| :------:    | :------:             | :------:             | :------:    | :------: |
| U2-Net bs1  | maxF:95.1% MAE:0.033 | maxF:94.8% MAE:0.033 | 111.147 fps | 254 fps  |
| U2-Net bs16 | maxF:95.1% MAE:0.033 | maxF:94.8% MAE:0.033 | 141.465 fps | 227 fps  |
| U2-Net bs4  | maxF:95.1% MAE:0.033 | -                    | 136.839 fps | 236 fps  |
| U2-Net bs8  | maxF:95.1% MAE:0.033 | -                    | 137.433 fps | 229 fps  |
| U2-Net bs32 | maxF:95.1% MAE:0.033 | -                    | 138.925 fps | 224 fps  |
| U2-Net bs64 | maxF:95.1% MAE:0.033 | -                    | 135.058 fps | 217 fps  |
|-------------|----------------------|----------------------|-------------|----------|

最优bs比较：
310P/T4=254/141.465=1.795，符合要求
