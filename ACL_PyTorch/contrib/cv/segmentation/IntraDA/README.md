# IntraDA-deeplabv2模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```
说明：PyTorch选用开源1.8.0版本


2.获取，安装开源模型代码  
安装IntraDA
```shell
git clone https://github.com/feipan664/IntraDA.git -b master
cd IntraDA
git reset --hard 070b0b702fe94a34288eba4ca990410b5aaadc4a
pip3.7 install -e ./ADVENT
cd ..
```


3.获取权重文件  
[deeplabv2 InTraDA训练的pth权重文件](https://z1hqmg.dm.files.1drv.com/y4mWPCqb6XAb6jMyXF82UaWj8kkUvBaUydSOdglID3YB_r1dolC0cc3-cWlI5RH7cN5PuKf96Bqg3e366STMbXLqpGmze8-gXy7Lq71OEAEx7ZSHjp9wNIIPNCNxE2F1s6u8xRxClXO2K5Tbs8Tde4hiGSm119pGKmn3W8X4U7IQZa8KS5lS5BztIOKkqEH1hZol7ELwbqm7i5TRzPPIMsycg)

4.数据集     
[CityScape](https://www.cityscapes-dataset.com/downloads/) 下载其中gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip两个压缩包，并解压。


5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/
```
 **评测结果：**   
| 模型      | 在线推理精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| intraDA-deeplabv2 bs1  | [mIoU：47.0](https://github.com/feipan664/IntraDA) | mIoU:47.0 |  27.396fps | 27.88fps | 
| intraDA-deeplabv2 bs16 | [mIoU：47.0](https://github.com/feipan664/IntraDA) | mIoU:47.0 | 30.81fps | 26.96fps | 



