# ResNeXt-50-32x4d 

This implements training of ResNeXt-50-32x4d on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet). 

## ResNeXt-50-32x4d Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, ResNeXt-50-32x4dV2 is re-implemented using semantics such as custom OP. For details, see models/main.py . 


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training 

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

# 1p training 1p
bash ./test/train_full_1p.sh  --data_path=xxx          # training accuracy

bash ./test/train_performance_1p.sh  --data_path=xxx   # training performance

# 8p training 8p
bash ./test/train_full_8p.sh  --data_path=xxx          # training accuracy

bash ./test/train_performance_8p.sh  --data_path=xxx   # training performance

# eval default 8p， should support 1p
bash ./test/train_eval_8p.sh  --data_path=xxx

# FP32 training with prof 
bash scripts/run_1p_prof.sh

# run demo.py
bash scripts/run_demo.sh

# run onnx
bash scripts/run_onnx.sh

## Traing log
test/output/devie_id/train_${device_id}.log                                  # training detail log

test/output/devie_id/ResNeXt-50-32x4d_${device_id}_bs_8p_perf.log            # 8p training performance result

test/output/devie_id/ResNeXt-50-32x4d_${device_id}_bs_8p_acc.log             # 8p training accuracy result

## ResNeXt-50-32x4dV2 training result 

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 597.527   | 1        |  1       | O2       |
| 77.726   | 2207.579  | 8        |  90      | O2       |

# Else 
