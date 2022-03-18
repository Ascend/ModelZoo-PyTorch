# RetinaNet(Detectron2)

## RetinaNet Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, RetinaNet is re-implemented using semantics such as custom OP. For details, see detectron2/modeling/meta_arch/retinanet.py


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- Install detectron2
    - Download RetinaNet from https://gitee.com/ascend/modelzoo.git
    - Then, cd contrib/PyTorch/Official/cv/image_object_detection/RetinaNet
    - Then, pip3.7 install -e .
- Download the ImageNet dataset from http://cocodataset.org/#home
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
- When do the demo, need to download a picture locally and name it input1.jpg
## Training 

Before to train, preparing R-50.pkl and config weight in the config yaml file.
To train a model, run `tools/train_net.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# 1p train 1p
bash ./test/train_full_1p.sh  --data_path=数据集路径

#  8p train 8p
bash ./test/train_full_8p.sh  --data_path=数据集路径

# 8p eval
bash ./test/train_eval_8p.sh  --data_path=数据集路径

# To ONNX
python3.7.5 pthtar2onnx.py
```




