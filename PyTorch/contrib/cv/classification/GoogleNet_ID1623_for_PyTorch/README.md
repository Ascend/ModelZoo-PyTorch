# GoogleNet 

This implements training of googlenet on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## GoogleNet Detail

Details, see ./googlenet.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py`or `main-8p.py` with the desired model architecture and the path to the ImageNet dataset:



# 1p training 1p
bash ./test/train_full_1p.sh  --data_path=xxx          # training accuracy

bash ./test/train_performance_1p.sh  --data_path=xxx   # training performance

# 8p training 8p
bash ./test/train_full_8p.sh  --data_path=xxx          # training accuracy

bash ./test/train_performance_8p.sh  --data_path=xxx   # training performance

# eval default 8p， should support 1p
bash ./test/train_eval_8p.sh  --data_path=xxx

# online inference demo 
python3.7.5 demo.py

# To ONNX
python3.7.5 pthtar2onnx.py

## Traing log
test/output/devie_id/train_${device_id}.log              # training detail log

test/output/devie_id/GoogleNet_${device_id}_bs_8p_perf.log            # 8p training performance result

test/output/devie_id/GoogleNet_${device_id}_bs_8p_acc.log             # 8p training accuracy result

## GoogleNet training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 515       | 1        | 150      | O2       |
| 69.807   | 4653      | 8        | 150      | O2       |
