# WDSR

This implements training of WDSR on the DIV2K_x2 dataset.
- Reference impkementation:
```
url=https://github.com/ychfan/wdsr
branch=master
commit_id=b78256293c435ef34e8eab3098484777c0ca0e10
```

## WDSR Detail

For Details, see src/models/wdsr.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- The DIV2k Dataset can be downloaded from Reference impkementation ([readme](https://github.com/ychfan/wdsr/blob/master/README.md)), find `DIV2K dataset: DIVerse 2K resolution high quality images as used for the NTIRE challenge on super-resolution @ CVPR 2017` link , download `Train Data (HR images)`, `Validation Data (HR images)`, `Train Data Track 1 bicubic downscaling x2 (LR images)`, `Validation Data Track 1 bicubic downscaling x2 (LR images)`.Move the datasets to directory ./data/DIV2K/



## Training

To train a model, run `trainer.py` with the desired model architecture and the path to the DIV2K dataset:

```bash
# 1p training full
# 备注： 目标精度34.75；验收精度35.3625
bash test/train_full_1p.sh 

# 1p train perf
bash test/train_performance_1p.sh 

# 1p testing
bash test/eval_1p.sh 

# 8p training full
# 备注： 目标精度34.75；验收精度33.7371
bash test/train_full_8p.sh 

# 8p train perf
bash test/train_performance_8p.sh 

# 8p testing
bash test/train_eval_8p.sh --pre_train_model=real_pre_train_model

# demo
# 请将要测试图片路径作为lr_image的参数传入，输出会在output_sr 文件夹
python3 demo.py --pre_train_model real_pre_train_model --lr_image 0801x2.png
```


## wdsr training result

| PSNR (dB)            | Npu_nums | Epochs   | AMP_Type |
| :------: | :------: | :------: | :------: |
| 35.3625                | 1        | 30       | O2       |
| 34.3368          | 8        | 30      | O2       |