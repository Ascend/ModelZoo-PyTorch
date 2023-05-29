# Inception_v3 

This implements training of inception_v3 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## Inception_v3 Detail

Details, see ./inception.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py`or `main-8p.py` with the desired model architecture and the path to the ImageNet dataset:



# 1p prefomance training 1p
bash test/train_performance_1p.sh  --data_path=/data/imagenet

# 8p prefomance training 8p
bash test/train_performance_8p.sh --data_path=/data/imagenet

# 1p full training 1p
bash test/train_full_1p.sh --data_path=/data/imagenet

# 8p full training 8p
bash test/train_full_8p.sh --data_path=/data/imagenet

# online inference demo 
python3 demo.py

# To ONNX
python3 pthtar2onnx.py

# 多机多卡性能数据获取流程
     ```
     1. 安装环境
     2. 开始训练，每个机器所请按下面提示进行配置
            bash ./test/train_performance_multinodes.sh  --data_path=数据集路径 --batch_size=单卡batch_size*单机卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```

## Inception_v3 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 295       | 1        | 250      | O2       |
| 78.4859   | 3251      | 8        | 240      | O2       |
