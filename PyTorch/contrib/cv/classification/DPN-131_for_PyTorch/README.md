# DPN-131

This implements training of ResNet18 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

# DPN-131 Detail

## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)


## Training

- 1p training 1p 
    - bash ./test/train_full_1p.sh --data_path=xxx # training accuracy

    - bash ./test/train_performance_1p.sh --data_path=xxx # training performance

- 8p training 8p
    - bash ./test/train_full_8p.sh --data_path=xxx # training accuracy
    
    - bash ./test/train_performance_8p.sh --data_path=xxx # training performance

- eval default 8p， should support 1p
    - bash ./test/train_eval_8p.sh --data_path=xxx

- Traing log
    - test/output/devie_id/train_${device_id}.log # training detail log
    
    - test/output/devie_id/ResNet101_${device_id}_bs_8p_perf.log # 8p training performance result
    
    - test/output/devie_id/ResNet101_${device_id}_bs_8p_acc.log # 8p training accuracy result    

### O2 online inference demo
source scripts/set_npu_env.sh
python3 demo.py

## Res2Net101_v1b training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 147       | 1        | 1        | O2       |
| 79.089   | 1129      | 8        | 120      | O2       |
