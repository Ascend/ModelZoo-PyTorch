[TOC]

### 文件作用说明：

- inf.sh                             // 用于性能测试脚本
- run.sh                           // 用于精度测试脚本
- acl_net.py                     // ACL接口脚本
- atc.sh                             // 转om模型脚本
- env.sh                            // 设置atc环境变量脚本
- export_onnx.py           // 导出模型脚本
- vqa-vilbert_bs1.om     // om模型文件
- vilbert-vqa-pretrained.2021-03-15.tar.gz                 // VilBert预训练权重
- bert-base-uncased/pytorch_model.bin                    // bert-base预训练权重
- feature_cache/                                                              // 图像特征预处理缓存所在文件夹
- fasterrcnn_resnet50_fpn_coco-258fb6c6.pth          // fasterrcnn预训练权重
- vqa-vilbert_bs1.onnx                                                   // onnx模型文件
- README.md

---

### 推理端到端步骤：

#### 1. 安装allennlp和allennlp-models
```shell
pip3.7 install allennlp==2.1.0 allennlp-models==2.1.0
```

#### 2. 下载VQA v2数据集
这里使用的是其中的Balanced Real Images数据集，下载连接：https://visualqa.org/download.html
仅下载**Validation images**（其它部分会自动下载），将下载的数据集解压到val2014目录，运行以下命令：
```shell
mkdir -p /net/nfs2.allennlp/data/vision/vqa
ln -s val2014_realpath /net/nfs2.allennlp/data/vision/vqa/balanced_real
```
其中的***val2014_realpath***填写你val2014图像数据压后的完整路径

#### 3. 将模型包中的文件上传到服务器某一目录，运行命令打上patch

```shell
python3.7 patch.py
```

#### 4. 下载网络权重并导出onnx（需要联网）

- 下载vilbert-vqa-pretrained.2021-03-15.tar.gz权重，放置于models目录下，下载连接：https://storage.googleapis.com/allennlp-public-models/vilbert-vqa-pretrained.2021-03-15.tar.gz
- 下载faster-rcnn预训练模型，放置于/root/.cache/torch/hub/checkpoints目录下，下载连接：https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
- 下载bert-base-uncased预训练权重，放置于根目录下的bert-base-uncased目录，下载连接：https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin

**注意：当网络环境不稳定时，上面几步可能出现ssl verify错误，修改python的requests包中的adapters.py文件，例如：/usr/local/python3.7.5/lib/python3.7/site-packages/requests/adapters.py
在HTTPAdapter类的send()函数开头加入一行verify=False**

- 运行以下命令导出onnx：
```shell
python3.7 export_onnx.py
```

#### 5. 简化模型

   对导出的onnx模型使用onnx-simplifer工具进行简化，将模型中shape固定下来，以提升性能。

   执行命令：

```shell
python3.7 -m onnxsim models/vqa-vilbert_bs1.onnx models/vqa-vilbert_bs1_sim.onnx
```

#### 6. 修改模型。

   进入om_gener目录，执行以下命令安装改图工具。
```shell
pip3.7 install .
```

   对模型进行修改，执行脚本。

```shell
python3.7 modify_vbt.py models/vqa-vilbert_bs1_sim.onnx
```

#### 7. 执行atc.sh脚本，将.onnx文件转为离线推理模型文件.om文件。

```shell
bash atc.sh
```

#### 8. 开始推理验证
为了节省数据预处理时间，将提供的feature_cache目录下的文件拷贝到/net/nfs2.allennlp/akshitab/data/vision/vqa/balanced_real/feature_cache_dir路径下

执行om离线推理命令：
```shell
bash run.sh
```

得到推理结果如下：
```shell
precision: 0.99, recall: 0.48, fscore: 0.65, vqa_score: 0.89, loss: 1.62 ||: :214354it [1:58:00, 30.27it/s]
```

#### 9. 获取推理性能
使用benchmark工具进行推理：
```shell
./benchmark -om_path=./vqa-vilbert_bs1.om -batch_size=1 -round=10 -device_id=0
```
得到性能数据如下：
```shell
[INFO] ave_thoroughputRate: 39.546samples/s, ave_latency: 25.4326ms
```