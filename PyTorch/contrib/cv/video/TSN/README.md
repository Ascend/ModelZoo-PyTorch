# TSN(mmaction2)

This implements training of TSN on the UCF101 dataset, mainly modified from [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2).

## TSN Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, TSN is re-implemented using semantics such as custom OP.


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- pip install -r requirements.txt
- Download the UCF101 dataset according to dataset/README.md

## Build MMCV
Download mmcv form source
```
git clone -b v1.3.9 https://github.com/open-mmlab/mmcv.git
mv mmcv/ mmcv-master/
mv mmcv-master/mmcv ./
rm -rf mmcv-master/
```

Change file of mmcv 
```
/bin/cp -f mmcv_need/base_runner.py mmcv/runner/base_runner.py
/bin/cp -f mmcv_need/builder.py mmcv/runner/optimizer/builder.py
/bin/cp -f mmcv_need/checkpoint.py mmcv/runner/hooks/checkpoint.py
/bin/cp -f mmcv_need/data_parallel.py mmcv/parallel/data_parallel.py
/bin/cp -f mmcv_need/dist_utils.py mmcv/runner/dist_utils.py
/bin/cp -f mmcv_need/distributed.py mmcv/parallel/distributed.py
/bin/cp -f mmcv_need/epoch_based_runner.py mmcv/runner/epoch_based_runner.py
/bin/cp -f mmcv_need/iter_timer.py mmcv/runner/hooks/iter_timer.py
/bin/cp -f mmcv_need/optimizer.py mmcv/runner/hooks/optimizer.py
/bin/cp -f mmcv_need/test.py mmcv/engine/test.py
/bin/cp -f mmcv_need/transformer.py mmcv/cnn/bricks/transformer.py
```

## Training 

To train a model, run `train.py` with the desired model architecture.

```bash
# 1p train perf
bash test/train_performance_1p.sh --data_path=/opt/npu

# 8p train perf
bash test/train_performance_8p.sh --data_path=/opt/npu

# 8p train full
bash test/train_full_8p.sh --data_path=/opt/npu

# 8p eval 
bash test/train_eval_8p.sh --data_path=/opt/npu

# finetuning
bash test/train_finetune_1p.sh --data_path=/opt/npu

# online inference demo
source test/env_npu.sh
python3 demo.py
```
Note:
- If you save dataset in another path but not /opt/npu, please specify argument --data_path.
For example, if your dataset path is /home/dataset/ucf101, then --data_path=/home/dataset.
- You can modify train_finetune_1p.sh to use another pretrain model.
- You can use argument --test_num to choose which video to be tested when using demo.py.

Log Path:
- tsn_performance_1p.log    # 1p Training performance result log
- tsn_performance_8p.log    # 8p Training performance result log
- tsn_full_1p.log       # 1p Training performance and accuracy result log
- tsn_full_8p.log       # 8p Training performance and accuracy result log
- tsn_eval_8p.log       # 8p validating accuracy result log
- tsn_finetune_1p.log   # 1p fine-tuning result log

## TSM training result 

| Top1 acc |   FPS   |  Epochs | AMP_Type |  Device  |
|  :---:   | :-----: |  :---:  | :------: | :------: |
|    -     |  111.82  |    1    |    O2    |  1p Npu  |
|  83.27   | 638.85 |    75   |    O2    |  8p Npu  |
|    -     |  59.88  |    1    |    O2    |  1p Gpu  |
|  82.29   | 958.66 |    75   |    O2    |  8p Gpu  |

# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md