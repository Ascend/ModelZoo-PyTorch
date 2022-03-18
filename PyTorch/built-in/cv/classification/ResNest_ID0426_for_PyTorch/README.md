# ResNEsts and DenseNEsts: Block-based DNN Models with Improved Representation Guarantees

This repository is the official implementation of the empirical research presented in the supplementary material of the paper, [ResNEsts and DenseNEsts: Block-based DNN Models with Improved Representation Guarantees](#BibTex).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> Please install Python before running the above setup command. The code was tested on Python 3.8.10.

Create a folder to store all the models and results:
```
mkdir ckeckpoint
```

## Training

To fully replicate the [results](#Results) below, train all the models by running the following two commands:

```replicate0
./train_cuda0.sh
```

```replicate1
./train_cuda1.sh
```

We used two separate scripts because we had two NVIDIA GPUs and we wanted to run two training processes for different models at the same time. If you have more GPUs or resources, you can submit multiple jobs and let them run in parallel.

To train a model with different seeds (initializations), run the command in the following form:

```train_and_test
python main.py --data <dataset> --model <DNN_model> --mu <learning_rate>
```

> The above command uses the default seed list. You can also specify your seeds like the following example:
```train_and_test_seed
python main.py --data CIFAR10 --model CIFAR10_BNResNEst_ResNet_110 --seed_list 8 9
```

Run this command to see how to customize your training or hyperparameters:
```options
python main.py --help
```

## Evaluation

To evaluate all trained models on benchmarks reported in the [tables](#Results) below, run:
```eval_all
./eval.sh
```

To evaluate a model, run:

```eval
python eval.py --data  <dataset> --model <DNN_model> --seed_list <seed>
```

## Results

### Image Classification on CIFAR-10

| Architecture | Standard | ResNEst | BN-ResNEst | A-ResNEst |
| ------------ |------------ |------------ |------------ |------------ |
| WRN-16-8 | 95.58% (11M) | 94.47% (11M) | 95.49% (11M) | 95.29% (8.7M) |
| WRN-40-4 | 95.49% (9.0M) | 94.64% (9.0M) | 95.62% (9.0M) | 95.48% (8.4M) |
| ResNet-110 | 94.33% (1.7M) | 92.62% (1.7M) | 94.47% (1.7M) | 93.93% (1.7M) |
| ResNet-20 | 92.58% (0.27M) | 90.98% (0.27M) | 92.56% (0.27M) | 92.47% (0.24M) |

### Image Classification on CIFAR-100

| Architecture | Standard | ResNEst | BN-ResNEst | A-ResNEst |
| ------------ |------------ |------------ |------------ |------------ |
| WRN-16-8 | 79.14% (11M) | 75.42% (11M) | 78.98% (11M) | 78.74% (8.9M) |
| WRN-40-4 | 79.08% (9.0M) | 75.16% (9.0M) | 78.81% (9.0M) | 78.69% (8.7M) |
| ResNet-110 | 74.08% (1.7M) | 69.08% (1.7M) | 74.24% (1.7M) | 72.53% (1.9M) |
| ResNet-20 | 68.56% (0.28M) | 64.73% (0.28M) | 68.49% (0.28M) | 68.16% (0.27M) |

## BibTeX
```
@inproceedings{chen2021resnests,
  title={{ResNEsts} and {DenseNEsts}: Block-based {DNN} Models with Improved Representation Guarantees},
  author={Chen, Kuan-Lin and Lee, Ching-Hua and Garudadri, Harinath and Rao, Bhaskar D.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

