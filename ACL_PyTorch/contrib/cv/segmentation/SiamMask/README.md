# SiamMask模型PyTorch离线推理指导

## 1 文件结构

```
.
├── LICENSE
├── modelzoo_level.txt
├── SiamMask.patch
├── SiamMask_eval.py
├── SiamMask_pth2onnx.py
├── README.md
├── requirements.txt
├── test
│   ├── eval_acc_perf.sh
│   ├── parse.py
│   └── pth2om.sh
└── SiamMask_test.py
```

## 2 环境准备

1. #### 安装必要的依赖

   ```
   pip3.7 install -r requirements.txt
   ```

2. #### 获取，修改与安装开源模型代码

   ```shell
   git clone https://github.com/foolwood/SiamMask.git -b master 
   cd SiamMask
   git reset 0eaac33050fdcda81c9a25aa307fffa74c182e36 --hard
   
   cd utils/pyvotkit
   python3.7 setup.py build_ext --inplace
   cd ../../
   
   cd utils/pysot/utils/
   python3.7 setup.py build_ext --inplace
   cd ../../../../
   ```

3. #### 获取权重文件

   下载权重文件SiamMask_VOT.pth，放到当前目录。
   
   ```
   wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
   ```
   
4. #### 获取VOT2016数据集，参见[Testing->Download test data](https://github.com/foolwood/SiamMask)

   请参考脚本中的命令构造数据集结构，如果数据集已存放到固定路径，可以通过ln -s软链接到`./SiamMask/data/VOT2016`。

5. #### 获取benchmark工具和msame工具

   将benchmark.x86_64或benchmark.aarch64放到当前的目录。

   获取[msame](https://gitee.com/ascend/tools/tree/master/msame#https://gitee.com/ascend/tools.git)并编译出可执行文件，放到当前目录。

## 3 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲。

```
bash test/pth2om.sh
bash test/eval_acc_perf.sh
```


## 4 评测结果

|   模型   |                  官网pth精度                  | 310离线推理精度 | 基准性能 |  310性能  |
| :------: | :-------------------------------------------: | :-------------: | :------: | :-------: |
| SiamMask | [0.433](https://github.com/foolwood/SiamMask) |      0.427      | 91.41FPS | 302.35FPS |

### 备注

- msame保存为bin格式并以float32加载比TXT格式精度高一些，性能也更好。
- 由于pytorch与onnx框架实现引起的bn算子属性epsilon，momentum在两种框架下略微不同，onnx精度与pth精度差了一点，但是om与onnx精度一致。
- 因为SiamMask是前后帧连续处理，即上一帧的输入作为下一帧的输出，在Refine模块中进行动态pad，因此模型需要拆分为两段。因为SiamMask部分卷积存在使用自定义kernel来对输入进行卷积操作，导致卷积存在kernel和input的双输入的情况，[issue](http://github.com/onnx/onnx-tensorrt/issues/645)，故在线推理测试性能。计算的是拆分的两部分模型合在一起完整推理的性能。
- 因为SiamMask在corr部分固定了reshape之后的形状，并且针对前后帧连续处理，所以模型不支持多batch。
- 推理速度较慢的问题目前出在io部分。如果`msame`支持将数据从内存中读写而不必须从`.bin`文件中读写，速度将会进一步加快。
