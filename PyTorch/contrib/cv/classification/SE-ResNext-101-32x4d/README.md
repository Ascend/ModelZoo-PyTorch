# SE-ResNeXt101_32x4d

This implements training of SE-ResNeXt101_32x4d on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/models/seresnext101_32x4d.py)

## SE-ResNeXt101_32x4d Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, SE-ResNeXt101_32x4d is re-implemented using semantics such as custom OP.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash 
# training 1p accuracy
bash test/train_full_1p.sh --data_path=/opt/npu/imagenet

# training 1p performance
bash test/train_performance_1p.sh --data_path=/opt/npu/imagenet

# training 8p accuracy
bash test/train_full_8p.sh --data_path=/opt/npu/imagenet

# training 8p performance
bash test/train_performance_8p.sh --data_path=/opt/npu/imagenet

#test 8p accuracy
bash test/train_eval_8p.sh --data_path=/opt/npu/imagenet --pth_path="./checkpointmodel_best.pth"

# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=/opt/npu/imagenet --pth_path="checkpointmodel_best.pth"
```

Log path:

```bash 
test/output/devie_id/train_${device_id}.log           # training detail log
test/output/devie_id/Se-ResNext101_bs1024_8p_perf.log  # 8p training performance result log
test/output/devie_id/Se-ResNext101_bs1024_8p_acc.log   # 8p training accuracy result log
```

## online inference demo
`python3 demo.py`


## SE-ResNeXt101_32x4d training result

| Acc@1    | FPS       | Platform| Device_nums| Epochs   | Type     |
| :------: | :------:  | :------ | :------    | :------: | :------: |
|  -       |  221      | GPU     | 1          |   1      | O2       |
|  -       |  395      | NPU     | 1          |   1      | O2       |
|  78.34  |  1480    | GPU     | 8          | 100      | O2       |
|  77.75  |  1978     | NPU     | 8          | 100      | O2       |