Deep Learning Recommendation Model for Personalization and Recommendation Systems:
=================================================================================
Description:
------------
An implementation of a deep learning recommendation model (DLRM)
The model input consists of dense and sparse features. The former is a vector
of floating point values. The latter is a list of sparse indices into
embedding tables, which consist of vectors of floating point values.
The selected vectors are passed to mlp networks denoted by triangles,
in some cases the vectors are interacted through operators (Ops).

Reference link：https://github.com/facebookresearch/dlrm

Requirements
------------
pytorch-nightly (*11/10/20*)

scikit-learn

numpy

npu torch

mlperf-logging
（You need to download mlperf-logging in this way “git clone https://github.com/mlperf/logging.git mlperf-logging”

“pip install -e mlperf-logging”）

onnx (*optional*)

pydot (*optional*)

torchviz (*optional*)

mpi (*optional for distributed backend*)

## Training

```
Before starting training you should download the dataset to the newly created data file
# 1p train 1p
bash ./test/train_full_1p.sh  --data_path=./data # train accuracy

bash ./test/train_performance_1p.sh --data_path=./data # train performance

# 1p eval 1p
bash test/train_eval_1p.sh --data_path=./data
```