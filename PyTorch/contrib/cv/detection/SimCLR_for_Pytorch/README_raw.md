# SimCLR_for_Pytorch

This project enables the SimCLR model could be trained on NPU, and remains the similar precision compared to the results
of the GPU.


## Requirements

- NPU配套的run包安装（建议安装 20.2.0.rc1 版本，请用以下脚本确认版本号，不确保其他版本能正常训练/测评）

  ```sh
  ll /usr/local/Ascend/ascend-toolkit/latest
  ```

- Python v3.7.5
- PyTorch v1.5 (NPU版本)
- Apex (NPU版本)


## Training

To train a model, run `xxx.sh` with the desired model architecture and the path to the CIFAR10 dataset:

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path
```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log


## SimCLR Training Results

| Acc@1    | FPS        | # of NPU/GPU | Epochs   | Opt-Level | Loss Scale |
| :------: | :------:   | :------:     | :------: | :------:  | :------:   |
| ------   | 1767.030   | 1P GPU       | 100      | O2        | 128.0      |
| 60.352   | 2098.001   | 1P NPU       | 100      | O2        | 128.0      |
| 55.859   | 5227.504   | 8P GPU       | 100      | O2        | 128.0      |
| 58.594   | 9747.414   | 8P NPU       | 100      | O2        | 128.0      |
