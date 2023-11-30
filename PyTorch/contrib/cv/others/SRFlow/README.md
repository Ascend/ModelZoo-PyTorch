# SRFlow

This implements training of SRFlow on the [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset, mainly modified from [sanghyun-son/EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch).


## SRFlow Detail

Details, see code/models/SRFlow_model.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
   Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download the DIV2K dataset from http://data.vision.ee.ethz.ch/alugmayr/SRFlow/datasets.zip (~10.7 GB)，Unzip to any directory.
- Download the pretrained_models from http://data.vision.ee.ethz.ch/alugmayr/SRFlow/pretrained_models.zip, Unzip to the SRFlow directory.


## Training

To train a model, run `train.py` with the desired model architecture and the path to the DIV2K dataset:

```bash
# xxx is the decompressed directory of datasets.zip, such as /home/datasets
# 1p train perf
bash test/train_performance_1p.sh --data_path xxx

# 8p train perf
bash test/train_performance_8p.sh --data_path xxx

# 8p train full
# Remarks: Target accuracy 23.05; test accuracy 23.98
# Save the model to experiments/train/models/latest_G.pth
bash test/train_full_8p.sh --data_path xxx 

# 1p eval
# The log file of performance and accuracy is output correctly
bash test/train_eval_1p.sh --data_path xxx 

# finetuning
bash test/train_finetune_1p.sh --data_path xxx 

# online inference demo 
# The prediction results are output correctly, and the output results of multiple runs of the fixed tensor are consistent
python3 demo.py
```


## SRFlow training result
|  名称  | 精度  | 性能 | AMP_Type |
| :----: | ----- | ---- | -------- |
| NPU-1p | -     | 10.3 | O1       |
| NPU-8p | 23.98 | 56.2 | O1       |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md