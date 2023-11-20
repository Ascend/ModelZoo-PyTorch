# Single-Stage Semantic Segmentation from Image Labels

please download data from origin repo:
https://github.com/visinf/1-stage-wseg/tree/master/data

### Setup
1. The training requires at least two Titan X GPUs (12Gb memory each).
2. **Setup your Python environment.** Please, clone the repository and install the dependencies.
    ```
    pip install -r requirements.txt
    ```
3. **Download and link to the dataset.** We train our model on the original Pascal VOC 2012 augmented with the SBD data (10K images in total). Download the data from:
    - VOC: [Training/Validation (2GB .tar file)](https://www.kaggle.com/a1173161983/dataset-for-wseg-vocsbd?select=voc)
    - SBD: [Training (1.4GB .tgz file)](https://www.kaggle.com/a1173161983/dataset-for-wseg-vocsbd?select=sbd)

    Make sure that the first directory in `real_data_path/voc` is `VOCdevkit`; the first directory in `real_data_path/sbd` is `benchmark_RELEASE`.
    in sbd ,you also should change the name `cls` to `cls_png`. finally the directory is:  ./real_data_path/voc/VOCdevkit/VOC2012/...; ./real_data_path/sbd/benchmark_RELEASE/dataset/...
4. **Download pre-trained models.** Download the initial weights (pre-trained on ImageNet) for the backbones you are planning to use and place them into `<project>/models/weights/`.

    | Backbone | Initial Weights |
    |:---:|:---:|
    | WideResNet38 | [ilsvrc-cls_rna-a1_cls1000_ep-0001.pth (402M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/models/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth)  |
    

## Training

To train a model, run `train_1P_NPU.py` or `train_8P_NPU.py` with the desired model architecture and the path to the sbd dataset:
```bash
# training 1p accuracy
bash ./test/train_full_1P.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1P.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8P.sh --data_path=real_data_path

# training 8p performance 
bash ./test/train_performance_8P.sh --data_path=real_data_path

#test 8p accuracy
bash test/train_eval_8P.sh --data_path=real_data_path --pth_path=real_pre_train_model_path

# finetuning 1p 
bash test/train_finetune_1P.sh --data_path=real_data_path --pth_path=real_pre_train_model_path
```


## Wseg training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 4.5       | 1        | 24        | O1       |
| 57.0 | 32 | 8        | 24      | O1       |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md