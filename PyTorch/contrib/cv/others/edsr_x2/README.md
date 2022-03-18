# EDSR_x2

This implements training of EDSR_x2 on the [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset, mainly modified from [sanghyun-son/EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch).


## EDSR_x2 Detail

Details, see src/model/edsr.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the DIV2K dataset from https://cv.snu.ac.kr/research/EDSR/DIV2K.tar (~7.1 GB). (训练时，会将数据集打包成bin文件，因此总的数据集大小会增加)


## Training

To train a model, run `main.py`or `main-8p.py` with the desired model architecture and the path to the DIV2K dataset:

```bash
# real_data_path为包含DIV2K数据集文件夹的目录

# 1p training full
# 备注： 目标精度35.03；验收精度35.001
bash test/train_full_1p.sh --data_path=real_data_path

# 1p train perf
bash test/train_performance_1p.sh --data_path=real_data_path

# 1p testing
bash test/train_eval_1p.sh.sh --pre_train_model=/path/to/model_best.pt --data_path=real_data_path

# 8p training full
# 备注： 目标精度35.03；验收精度34.9426
bash test/train_full_8p.sh --data_path=real_data_path

# 8p train perf
bash test/train_performance_8p.sh --data_path=real_data_path

# 8p testing
bash test/train_eval_8p.sh.sh --pre_train_model=/path/to/model_best.pth --data_path=real_data_path

# finetuning
bash test/train_finetune_1p.sh --data_path=real_data_path --pre_train_path=/path/to/model_best.pt

# demo
# 先将待测试图片放到 test 文件夹，输出图片会放在在 output_sr 文件夹
python3.7.5 demo.py --cpu --pre_train=/path/to/model_best.pt_OR_pth
```


## EDSR_x2 training result

| PSNR (dB)            | Npu\_nums | Epochs   | AMP\_Type |
| :------: | :------: | :------: | :------: |
| 35.001                | 1        | 281        | O2       |
| 34.9426          | 8        | 300      | O2       |

