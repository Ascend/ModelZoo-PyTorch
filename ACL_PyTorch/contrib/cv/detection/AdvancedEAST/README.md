# AdvancedEAST模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖
```
pip3.7 install -r requirements.txt
```

2.获取开源模型代码
```
git clone https://github.com/BaoWentz/AdvancedEAST-PyTorch -b master
cd AdvancedEAST-PyTorch
git reset a835c8cedce4ada1bc9580754245183d9f4aaa17 --hard
cd ..  
```

3.获取权重文件

[AdvancedEAST预训练pth权重文件](https://pan.baidu.com/s/1NSyc-cHKV3IwDo6qojIrKA)，密码: ye9y

解压后使用3T736_best_mF1_score.pth，文件sha1: 9D0C603C4AA4E955FEA04925F3E01E793FEF4045

4.获取数据集

[天池ICPR数据集](https://pan.baidu.com/s/1NSyc-cHKV3IwDo6qojIrKA)，密码: ye9y

下载ICPR_text_train_part2_20180313.zip和[update] ICPR_text_train_part1_20180316.zip两个压缩包，新建目录icpr和子目录icpr/image_10000、icpr/txt_10000，将压缩包中image_9000、image_1000中的图片文件解压至image_10000中，将压缩包中txt_9000、txt_1000中的标签文件解压至txt_10000中

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)

将benchmark.x86_64或benchmark.aarch64放到当前目录

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲
```
bash test/pth2om.sh
bash test/eval_acc_perf.sh
```
**评测结果：**
| 模型      | 官网pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  |
| AdvancedEAST bs1  | f1-score:52.08% | f1-score:52.08% | 84.626fps | 90.9036fps |
| AdvancedEAST bs16 | f1-score:52.08% | f1-score:52.08% | 86.304fps | 90.6276fps |

**备注：**
- Torch 1.5.0导出的onnx需要使用onnxsim并且onnx不支持动态batch，Torch 1.8.0不存在上述问题，因此Torch选用1.8.0。
- 未使用autotune前310 bs1/bs16的fps为81.2012/81.3156，使用autotune优化后310 bs1/bs16的fps为90.9036/90.6276，使用autotune后性能达标。