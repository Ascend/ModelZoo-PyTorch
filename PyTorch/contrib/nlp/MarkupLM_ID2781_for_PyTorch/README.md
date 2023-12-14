# MarkupLM

**Multimodal (text +markup language) pre-training for [Document AI](https://www.microsoft.com/en-us/research/project/document-ai/)**

## Introduction

MarkupLM is a simple but effective multi-modal pre-training method of text and markup language for visually-rich document understanding and information extraction tasks, such as webpage QA and webpage information extraction. MarkupLM achieves the SOTA results on multiple datasets. For more details, please refer to our paper:

[MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document Understanding](https://arxiv.org/abs/2110.08518)  Junlong Li, Yiheng Xu, Lei Cui, Furu Wei, [Preprint](https://github.com/microsoft/unilm/tree/master/markuplm#)

The overview of our framework is as follows:
<div align="center">
<img src="https://user-images.githubusercontent.com/45759388/142979309-9b3ba8ce-d76c-482a-8ded-f837037e9e81.PNG" width="100%" height="100%" />
</div>

And the core XPath Embedding Layer is as follows:
<div align="center">
<img src="https://user-images.githubusercontent.com/45759388/142979238-22bc4910-1236-4c72-9292-47d613c39daa.PNG" width="70%" height="70%" />
</div>

## Release Notes

******* New Nov 22th, 2021: Initial release of pre-trained models and fine-tuning code for MarkupLM *******

## Pre-trained Models

We pre-train MarkupLM on a subset of the CommonCrawl dataset.

| Name  | HuggingFace |
| - | - | 
| MarkupLM-Base | [microsoft/markuplm-base](https://huggingface.co/microsoft/markuplm-base) |
| MarkupLM-Large | [microsoft/markuplm-large](https://huggingface.co/microsoft/markuplm-large) |

An example might be ``model = markuplm.from_pretrained("microsoft/markuplm-base")``.

## Installation

### Command

```
conda create -n markuplmft python=3.7
conda activate markuplmft
git clone https://github.com/microsoft/unilm.git
cd unilm
cd markuplm
pip install -r requirements.txt
pip install -e .
```

## Finetuning

### SWDE

#### Prepare data

Download the dataset from the [official website](https://archive.codeplex.com/?p=swde).

Update: the above website is down, please use this [backup](http://web.archive.org/web/20210630013015/https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip).

Unzip **swde.zip**, and extract everything in **/sourceCode**, make sure we have folders like **auto / book / camera** ... under this directory, and we name this path as **/Path/To/SWDE**.

#### Generate dataset

```
cd ./examples/fine_tuning/run_swde

python pack_data.py \
	--input_swde_path /Path/To/SWDE \
	--output_pack_path /Path/To/SWDE/swde.pickle

python prepare_data.py \
	--input_groundtruth_path /Path/To/SWDE/groundtruth \
	--input_pickle_path /Path/To/SWDE/swde.pickle \
	--output_data_path /Path/To/Processed_SWDE
```

And the needed data is in **/Path/To/Processed_SWDE**.

#### Run

Take **seed=1, vertical=nbaplayer** as example.

```
python3 run.py --root_dir /home/test_user04/Path/To/Processed_SWDE --vertical nbaplayer --n_seed 1 --n_pages 2000 --prev_nodes_into_account 4 --model_name_or_path microsoft/markuplm-base --output_dir /home/test_user04/Your/Output/Path --do_train --do_eval --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --num_train_epochs 10 --learning_rate 2e-5 --save_steps 1000000 --warmup_ratio 0.1 --overwrite_output_dir --npu_id
```

### Results

#### SWDE
GPU
| Precision | recall | F1 |
|-------|-------|-------|
| 0.8864 | 0.8175 | 0.8571 |

NPU
| Precision | recall | F1 |
|-------|-------|-------|
| 0.9175 | 0.8426 | 0.8818 |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md
