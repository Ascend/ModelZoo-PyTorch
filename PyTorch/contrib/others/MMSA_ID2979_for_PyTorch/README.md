<h2 id="概述.md">概述</h2>

Pytorch implementation for codes in Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis (AAAI2021). Please see another repo MMSA for more details, which is a scalable framework for MSA.

- 参考论文：

    https://arxiv.org/abs/2102.04830v1

- 参考实现：

    https://github.com/thuiar/Self-MM


<h2 id="训练环境准备.md">训练环境准备</h2>

- Install PyTorch (pytorch.org)
- pip install -r requirements.txt
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download dataset features and pre-trained berts from the following link.
    
    [Baidu Cloud Drive](https://pan.baidu.com/s/1oksuDEkkd3vGg2oBMBxiVw) with code: `ctgs`
    
预训练权重文件路径
```
 ./pretrained_model/bert_cn/pytorch_model.bin
 ./pretrained_model/bert_en/pytorch_model.bin
```
数据集文件路径
```
 ./SIMS/Processed/features/SIMS-label.csv
 ./SIMS/Processed/features/unaligned_39.pkl
```

For all features, you can use SHA-1 Hash Value to check the consistency.
> `SIMS/unaligned_39.pkl`: `a00c73e92f66896403c09dbad63e242d5af756f8`  

Due to the size limitations, the SIMS raw videos are available in `Baidu Cloud Drive` only. All dataset features are organized as:

```
{
    "train": {
        "raw_text": [],
        "audio": [],
        "vision": [],
        "id": [], # [video_id$_$clip_id, ..., ...]
        "text": [],
        "text_bert": [],
        "audio_lengths": [],
        "vision_lengths": [],
        "annotations": [],
        "classification_labels": [], # Negative(< 0), Neutral(0), Positive(> 0)
        "regression_labels": []
    },
    "valid": {***}, # same as the "train" 
    "test": {***}, # same as the "train"
}
```
Make some changes
Modify the `config/config_tune.py` and `config/config_regression.py` to update dataset pathes.

<h2 id="模型训练.md">模型训练</h2>

```
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path
```
Log path: ./MA_LOG/MA-JOB-NEW-MA-new-Self-MM-xx-xx-xx-xx.log or obs://cann-idxxx/npu/workspace/MA-new-Self-MM-xxx/log/modelarts-job-xxx-worker-0.log

<h2 id="训练结果.md">训练结果</h2>

- 精度结果比对

|精度指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|ACC-2|0.807|0.786|0.784|
|F1-Score|0.808|0.785|0.784|
|MAE|0.419|0.421|0.417|
|Corr|0.616|0.587|0.593|
