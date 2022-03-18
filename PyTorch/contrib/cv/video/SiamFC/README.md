# SiamFC

This implements training of SiamFC on the ILSVRC2015-VID dataset, and testing of SiamFC on the OTB2015 dataset, mainly modified from [HonglinChu/SiamTrackers](https://github.com/HonglinChu/SiamTrackers/tree/master/2-SiamFC/SiamFC-VID).

## SiamFC Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. Therefore, SiamFC is re-implemented using semantics such as custom OP. For more details, see siamfc/train.py.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ILSVRC2015-VID dataset
- Download the OTB2015 dataset from [Visual Tracker Benchmark (hanyang.ac.kr)](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)

## Training

- Firstly, run bin/create_dataset.py and bin/create_lmdb.py to preprocess the ILSVRC2015-VID dataset. Be careful that:
  - real_data_path is where the ILSVRC2015-VID dataset is;
  - out_data_path is where the preprocessed images are;
  - lmdb_data_path must be out_data_path+".lmdb", 
    e.g. if out_data_path is "./data/ILSVRC_VID_CURATION", 
    then lmdb_data_path must be "./data/ILSVRC_VID_CURATION.lmdb";

```bash
python3.7 bin/create_dataset.py --d real_data_path --o out_data_path
python3.7 bin/create_lmdb.py --d out_data_path --o lmdb_data_path
```

- To train a model, run bin/my_train.py with the desired model architecture and out_data_path, which is mentioned above:


```bash
# training 1p performance
bash ./test/train_performance_1p.sh --data_path=out_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=out_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=out_data_path

# test 8p accuracy
bash ./test/train_eval_8p.sh --data_path=real_data_path
```

Log path:

    test/output/device_id/train_${device_id}.log       # training detail log
    test/output/device_id/siamfc_bs32_8p_fps_loss.log  # 8p training loss result log
    test/output/device_id/siamfc_bs32_8p_acc.log       # 8p training accuracy result log
    
    test/output/1p_perf/device_id/siamfc_bs32_1p_fps_loss.log  # 1p training performance result log
    
    test/output/8p_perf/device_id/siamfc_bs32_8p_fps_loss.log  # 8p training performance result log
    
    test/output/output.prof  # 1p training prof file




## SiamFC training result

| Precision |  Success  | FPS  | Npu_nums | Epochs | AMP_Type |
| :-------: | :-------: | :--: | :------: | :----: | :------: |
|     -     | - | 1730 |    1     |   1    |    O2    |
|  0.755  | 0.566 | 4842 |    8     |   50   |    O2    |
