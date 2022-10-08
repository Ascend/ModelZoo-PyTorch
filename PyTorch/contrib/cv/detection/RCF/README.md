# RCF

This implements training of RCF on the HED-BSDS_PASCAL dataset, mainly modified from (https://github.com/mayorx/rcf-edge-detection).

## RCF Detail

RCF is re-implemented using ResNet101.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  
    Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- dataset
    http://mftp.mmcheng.net/liuyun/rcf/data/bsds_pascal_train_pair.lst
    http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
    http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
    -下载BSDS500数据集(https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500)，将BSDS500/data/groundTruth/test/下的.mat文件保存到gt文件夹下
最后将所有文件保存到data/HED-BSDS_PASCAL

## Training

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/RCF_bs3_8p_perf.log  # 8p training performance result log
    test/output/devie_id/RCF_bs3_8p_perf.log   # 8p training accuracy result log



## WideResnet50_2 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 20.2       | 1        | 1        | O2       |
| 80.1   | 131.3      | 8        | 30      | O2       |
