# MSPN模型PyTorch离线推理指导

## 1 环境准备

1. 下载源代码

```
git clone https://github.com/megvii-research/MSPN -b master   
cd  MSPN 
```

1. 设置环境变量，将当前目录设置为程序运行的主目录

 ```
 export MSPN_HOME=$(pwd)
 export PYTHONPATH=$PYTHONPATH:$MSPN_HOME
 ```

3. 配置环境要求

 ```
 pip3 install -r requirements.txt
 ```

4. 下载COCOAPI

 ```
 git clone https://github.com/cocodataset/cocoapi.git $MSPN_HOME/lib/COCOAPI
 cd $MSPN_HOME/lib/COCOAPI/PythonAPI
 make install
 ```

5. 下载数据集

（1）下载COCO2014数据集[COCO website][1], 将 train2014/val2014文件夹分别放在“$MSPN_HOME/dataset/COCO/images/” 目录下.

（2）下载测试结果[Detection result][2], 把它放在“ $MSPN_HOME/dataset/COCO/det_json/”目录下.

（3）将预训练好的权重文件 “mspn_2xstg_coco.pth”放在当前目录

6.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer),将benchmark.x86_64或benchmark.aarch64放到当前工作目录

## 2 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```
bash test/pth2om.sh  
bash test/test.sh 
bash test/performance_test.sh
```

**评测结果：**

| 模型      | 官网pth精度 | 310离线推理精度 | 基准性能   | 310性能    |
| --------- | ----------- | --------------- | ---------- | ---------- |
| MSPN bs1  | 74.5        | 74.1            | 372.45fps  | 401.74fps  |
| MSPN bs16 | 74.5        | 74.1            | 712.271fps | 436.868fps |

## 相关链接

[1]: http://cocodataset.org/#download
[2]: https://drive.google.com/open?id=1MW27OY_4YetEZ4JiD4PltFGL_1-caECy

