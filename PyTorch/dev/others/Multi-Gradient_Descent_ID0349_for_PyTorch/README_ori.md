# MAMO Framework

> A Model Agnostic Multi-Objective Framework for Deep Learning models


[![Build Status](http://img.shields.io/travis/badges/badgerbadgerbadger.svg?style=flat-square)](https://travis-ci.org/badges/badgerbadgerbadger)

This framework will enable users to easily create, train, test and deploy deep learning models with a focus on multi-objective. The framework is easy to use and understand. It is developed in "Lego fashion", meaning that we already supply out of the box models, loss function, metrics which can be easily used to build and train models, but also we leave full flexibility to the user to define their own models, losses, metrics only by implementing abstract classes.  




---

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Architecture](#architecture)
- [Tests](#tests)
- [Team](#team)



---

## Example

This example is based on the [Multi-Gradient Descent for Multi-Objective Recommender Systems](https://arxiv.org/abs/2001.00846)(arXiv) paper.

```python
import torch
import numpy as np
import os

from dataloader.ae_data_handler import AEDataHandler
from models.multi_VAE import MultiVAE
from loss.vae_loss import VAELoss
from metric.recall_at_k import RecallAtK
from metric.revenue_at_k import RevenueAtK
from trainer import Trainer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/user/working_dir/data/", help="the path to the directory where the data is stored")
parser.add_argument("--models_dir", default="/home/user/working_dir/models", help="the path to the directory where to save the models, it must be empty")
args = parser.parse_args()

# get the arguments
dir_path = args.data_dir
save_to_path = args.models_dir

# set up logging
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# set npu if available
device = torch.device('npu' if torch.npu.is_available() else 'cpu')

train_data_path = os.path.join(
    dir_path, "movielens_small_training.npy")
validation_input_data_path = os.path.join(
    dir_path, "movielens_small_validation_input.npy")
validation_output_data_path = os.path.join(
    dir_path, "movielens_small_validation_test.npy")
test_input_data_path = os.path.join(
    dir_path, "movielens_small_test_input.npy")
test_output_data_path = os.path.join(
    dir_path, "movielens_small_test_test.npy")
products_data_path = os.path.join(
    dir_path, "movielens_products_data.npy")

data_handler = AEDataHandler(
    "MovieLensSmall", train_data_path, validation_input_data_path,
    validation_output_data_path, test_input_data_path,
    test_output_data_path)

input_dim = data_handler.get_input_dim()
output_dim = data_handler.get_output_dim()

products_data_np = np.load(products_data_path)
products_data_torch = torch.tensor(
    products_data_np, dtype=torch.float32).to(device)

# create model
model = MultiVAE(params="yaml_files/params_multi_VAE_training.yaml")

correctness_loss = VAELoss()
revenue_loss = VAELoss(weighted_vector=products_data_torch)
losses = [correctness_loss, revenue_loss]

recallAtK = RecallAtK(k=10)
revenueAtK = RevenueAtK(k=10, revenue=products_data_np)
validation_metrics = [recallAtK, revenueAtK]

trainer = Trainer(data_handler, model, losses, validation_metrics, save_to_path)
trainer.train()
print(trainer.pareto_manager._pareto_front)
```

### Downloading datasets
We provide two already preprocessed, ready to use datasets, and can be downloaded from the following links:
1. [MovieLens dataset with prices](https://drive.google.com/open?id=15KwO7tk9S4M5raro2ndkYswFLh7MpPkt).
2. [Amazon movies dataset with prices](https://drive.google.com/open?id=1O1XfAFxKAvUTXGTk6WQDO5H0OP9y5xuI).

Please, download and unzip the MovieLens dataset before continuing with the next steps.

### Running the example script
If all the steps are finished and all the requirements are satisfied, the script can be run with the following command:
```
python run_example.py --data_dir bar --models_dir foo
```
Where `--data_dir` is the path to the directory where the data downloaded in the previous step is uncompressed and stored, and `--models_dir` is the path to the directory where the framework will save the models belonging to the Pareto front and it must be empty directory.


---

## Installation

The MAMO framework is compatible with: Python 3.6+. PyTorch 1.2.0+.

### Clone

- Clone this repo using `https://git.swisscom.com/scm/dilrec/moframework.git`

### Dependencies

The dependencies needs to be satisfied in order to be able to use the MAMO Framework. There are several ways to install them.

> 1. Using pip and the requirements.txt configuration file provided

```shell
$ pip install -r requirements.txt
```

> 2. Downloading a [Nvidia PyTorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) Docker image, and run everything inside a Docker container. (preferred)

```shell
$ docker pull nvcr.io/nvidia/pytorch:18.12.1-py3
```

---

## Features
- Train models on multi-objectives.
- Flexibility for users to define their own model.
- Compatible with PyTorch models.
- Automatic model saving.
- YAML config files for model/data hyperparameters.

---



## Architecture
Our architecture is shown on the following diagram:

<a href="https://ibb.co/B4GjCZf"><img src="https://i.ibb.co/Q8knKNJ/MOMA-class-diagram.png" alt="MOMA-class-diagram" border="0"></a>


---

## Tests
The tests are written using [pytest](https://docs.pytest.org/en/stable/index.html). It needs to be installed if you want to run the tests.

In order to run our tests, you have to be in the `tests` directory and run them with `pytest`:
```
cd tests
pytest
```


---

## Team

Code written by the recommenders@swisscom, including:

[Kirtan Padh](https://github.com/kirtanp)

[Milena Filipovic](https://github.com/MilenaFilipovic)

[Nelson Antunes](https://github.com/Nelsi11120)

[Loic Nguyen](https://github.com/coiL10)

[Blagoj Mitrevski](https://github.com/blagojce95)

## 网络训练状况

FuncStatus:OK(流程通过)
PerfStatus:NOK(小于0.5倍GPU)
PrecisionStatus:POK(Loss拟合但精度未实施)
