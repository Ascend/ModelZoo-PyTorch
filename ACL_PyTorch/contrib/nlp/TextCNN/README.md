# Ascend-textcnn

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```
2.获取，修改与安装开源模型代码  

```
git clone https://gitee.com/zhang_kaiqi/ascend-textcnn.git -b master   
cd ascend-textcnn
git reset HEAD --hard
cd ..  
```
3.获取权重文件  

[TextCNN_9045_seed460473.pth](https://gitee.com/zhang_kaiqi/ascend-textcnn/blob/master/Chinese-Text-Classification-Pytorch/THUCNews/saved_dict/TextCNN_9045_seed460473.pth)  

4.数据集     
[test.txt](https://gitee.com/zhang_kaiqi/ascend-textcnn/tree/master/Chinese-Text-Classification-Pytorch/THUCNews/data)

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/doc/EDOC1100219269/84cbd58d)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  

```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```
 **评测结果：**   
|     模型     |                         官网pth精度                          | 310离线推理精度 | 基准性能 | 310性能 |
| :----------: | :----------------------------------------------------------: | :-------------: | :------: | :-----: |
| TextCNN bs1  | [91.22%](https://gitee.com/huangyd8/Chinese-Text-Classification-Pytorch) |     90.47%      | 5045fps  | 5568fps |
| TextCNN bs16 | [91.22%](https://gitee.com/huangyd8/Chinese-Text-Classification-Pytorch) |     90.47%      | 34271fps | 16456fps |

