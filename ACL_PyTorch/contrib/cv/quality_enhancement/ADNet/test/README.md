# 环境准备：

## 1.进入工作目录

cd ADNet

## 2.获取数据集

本模型支持BSD68数据集共68张数据集，可从百度云盘下载

链接：https://pan.baidu.com/s/1XiePOuutbAuKRRTV949FlQ 
提取码：0315

文件结构如下

```
|ADNet--test
|     |  |--pth2om.sh
|     |  |--perf_t4.sh
|     |  |--parse.py
|     |  |--eval_acc_perf.sh
|     |--datset
|     |  |--BSD68
|     |--prep_dataset
|     |  |--ISoure
|     |  |--INoisy
|     |--util.py
|     |--requirements.tx
|     |--models.py
|     |--gen_dataset_info.py
|     |--ADNet_pth2onnx.py
|     |--ADNet_preprocess.py
|     |--ADNet_postprocess.py
```



## 3.安装必要的依赖 ,测试环境可能已经安装其中的一些不同版本的库，故手动测试时不推荐使用该命令安装

pip3.7 install -r requirements.txt  

## 4.下载ADNet离线推理代码

下载ADNet推理代码

```
git clone https://gitee.com/wang-chaojiemayj/modelzoo.git
cd modelzoo
git checkout tuili
```

进入ADNet目录

```
cd ./contrib/ACL_PyTorch/Research/cv/quality_enhancement/ADnet
```



## 5.获取权重文件  

```  
wget https://github.com/hellloxiaotian/ADNet/blob/master/gray/g25/model_70.pth
```

## 6.获取benchmark工具  

将benchmark.x86_64 benckmark_tools放在当前目录  

## 7.310上执行，执行时确保device空闲  

bash test/pth2om.sh  
bash test/eval_acc_perf.sh 
