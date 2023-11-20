# MSPN

This implements training of WSPN on the COCO2014 dataset, mainly modified from [github][9].

## Repo Structure
This repo is organized as following:
```
$MSPN_HOME
|-- cvpack
|
|-- dataset
|   |-- COCO
|   |   |-- det_json
|   |   |-- gt_json
|   |   |-- images
|   |       |-- train2014
|   |       |-- val2014
|   |
|   |-- MPII
|       |-- det_json
|       |-- gt_json
|       |-- images
|   
|-- lib
|   |-- models
|   |-- utils
|
|-- exps
|   |-- mspn.2xstg.coco
|   
|---test
|	|--env_npu.sh
|	|--train_full_8p.sh
|	|--train_performance_1p.sh
|	|--train_performance_8p.sh
|
|-- model_logs
|
|-- README.md
|-- requirements.txt
```

## Quick Start

### Installation

1. Install Pytorch referring to [Pytorch website][2].

2. Clone this repo, and config **MSPN_HOME** in **/etc/profile** or **~/.bashrc**, e.g.
 ```
 export MSPN_HOME='/path/of/your/cloned/repo'
 export PYTHONPATH=$PYTHONPATH:$MSPN_HOME
 ```

3. Install requirements:
 ```
 pip3 install -r requirements.txt
 ```

4. Install COCOAPI referring to [cocoapi website][3], or:
 ```
 git clone https://github.com/cocodataset/cocoapi.git $MSPN_HOME/lib/COCOAPI
 cd $MSPN_HOME/lib/COCOAPI/PythonAPI
 make install
 ```

### Dataset

#### COCO

1. Download images from [COCO website][4], and put train2014/val2014 splits into **$MSPN_HOME/dataset/COCO/images/** respectively.

2. Download ground truth from [Google Drive][6], and put it into **$MSPN_HOME/dataset/COCO/gt_json/**.

3. Download detection result from [Google Drive][6], and put it into **$MSPN_HOME/dataset/COCO/det_json/**.

### Model
Download ImageNet pretained ResNet-50 model from [Google Drive][6], and put it into **$MSPN_HOME/lib/models/**. For your convenience, We also provide a well-trained 2-stage MSPN model for COCO.

### Log
Create a directory to save logs and models:
```
mkdir $MSPN_HOME/model_logs
```

### Train
Go to specified experiment repository, e.g.
```
# training 1p performance
bash ./test/train_performance_1p.sh 

# training 8p accuracy
bash ./test/train_full_8p.sh 

# training 8p performance
bash ./test/train_performance_8p.sh 
```
### Test
```
python -m torch.distributed.launch --nproc_per_node=gpu_num test.py -i iter_num
```
the ***gpu_num*** is the number of gpus, and ***iter_num*** is the iteration number you want to test. Remenber that test in gpu environment.

## MSPN training result

### Results on COCO val dataset 

| Acc@1 | FPS    | Npu_nums | Epochs | AMP_type |
| ----- | ------ | -------- | ------ | -------- |
| -     | 64.727 | 1        | 4      | O1       |
| 74.5  | 341.4  | 8        | 4      | O1       |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md