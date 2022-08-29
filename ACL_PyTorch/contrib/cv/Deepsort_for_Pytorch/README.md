# deepsort模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2. 获取，修改与安装开源模型代码  

```
git clone https://github.com/ZQPei/deep_sort_pytorch.git   
cd deep_sort_pytorch 
git reset 4c2d86229b0b69316af67d519f8476eee69c9b20 --hard
```

3. 下载权重并导出onnx

   ```
   cd detector/YOLOv3/weight/
   wget https://pjreddie.com/media/files/yolov3.weights
   cd ../../../
   ```

   下载第二段网络权重

   ```
   cd deep_sort/deep/checkpoint
   wget https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6
   cd ../../../
   ```

   也可从百度云下载：链接：https://pan.baidu.com/s/18GI_s87_eZu6ZdKx3eEqLQ 提取码：lwj1

   为后续导出onnx，可将其中一个视频文件下载，demo.avi放到deep_sort_pytorch文件夹下

   骨干网络有两段onnx, 第一段为yolov3，首先该网络返回的不是Tensor，需要修改detector/YOLOv3/darknet.py，该模型中使用三个anchor，将第150行修改为：

   ```
   return out_boxes[0]['x'], out_boxes[0]['a'], out_boxes[1]['x'], out_boxes[1]['a'], out_boxes[2]['x'], out_boxes[2]['a']
   ```

   在detector/YOLOv3/detector.py文件43行后添加如下代码：

   ```
   input_names = ["actual_input_1"]
   import onnx
   self.net.eval()
   torch.onnx.export(self.net, img, "yolov3.onnx", input_names = input_names, opset_version = 11)
   return
   ```

   以上代码修改也可以通过patch实现:

   ```shell
   cd deep_sort_pytorch
   patch -p1 < ../yolov3.patch
   ```
   注意检查一下环境。pyyaml的版本比5.1高的话需要在./deep_sort_pytorch/utils/parser.py的第24行替换为
   
   ```
   self.update(yaml.safe_load(fo.read()))
   ```

   运行如下导出onnx：

   ```
   python3.7.5 yolov3_deep_sort.py demo.avi --cpu
   ```

   导出onnx后即可删除该代码，导出的onnx包含动态shape算子where，当前无法支持，因此使用onnxsim进行常量折叠，消除动态shape，且该视频流仅支持batchsize=1的场景

   第二段网络为检测网络，将提供的export_deep_onnx.py脚本置于deep_sort_pytorch/deep_sort/deep路径下，运行该脚本，导出第二段onnx：

   ```
   python3.7.5 export_deep_onnx.py
   ```

4. 导出om离线模型
   
   ```
   bash atc.sh
   ```

## 2 推理验证

   评测时，首先删除detector/YOLOv3/detector.py中添加的导出onnx的代码

1. 替换文件

2. 

   原仓提供的评测脚本有误，使用我们提供的yolov3_deepsort_eval.py、并按照deepsort_revise_method.md修改对应detector.py、feature_extractor.py，也可以通过patch修改文件：

   ```shell
   cd deep_sort_pytorch
   patch -p1 < ../evaluation.patch
   ```

   

   并将acl_net_dynamic.py脚本放置在detector/YOLOv3/以及deep_sort/deep目录下，将yolov3-sim.om和deep_dims.om以及我们提供的yolov3_deepsort_eval.py放在deep_sort_pytorch目录下

   

3. 运行如下脚本即可获取精度

```
python3.7.5 yolov3_deepsort_eval.py
```



**评测结果：**   

|     模型     | 官网pth精度 | 310离线推理精度 | 310p精度 |             T4性能             |
| :----------: | :---------: | :-------------: | :-----: | :-----------------------------: |
| deepsort bs1 | MOTA:30.1  |   MOTA:30.1    |    MOTA:30.1     | yolov3:52.2FPS;deep_dims:619.9FPS |




