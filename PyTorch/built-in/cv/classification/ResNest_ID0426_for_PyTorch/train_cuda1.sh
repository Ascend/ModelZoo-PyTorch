#!/bin/bash
python3 main.py --data CIFAR10 --model CIFAR10_ResNEst_ResNet_110 --device cuda:1 --mu 0.01
python3 main.py --data CIFAR10 --model CIFAR10_ResNEst_ResNet_20 --device cuda:1 --mu 0.01
python3 main.py --data CIFAR10 --model CIFAR10_ResNEst_WRN_16_8 --device cuda:1 --mu 0.01
python3 main.py --data CIFAR10 --model CIFAR10_ResNEst_WRN_40_4 --device cuda:1 --mu 0.01
python3 main.py --data CIFAR10 --model CIFAR10_AResNEst_ResNet_110 --device cuda:1
python3 main.py --data CIFAR10 --model CIFAR10_AResNEst_ResNet_20 --device cuda:1
python3 main.py --data CIFAR10 --model CIFAR10_AResNEst_WRN_16_8 --device cuda:1
python3 main.py --data CIFAR10 --model CIFAR10_AResNEst_WRN_40_4 --device cuda:1
python3 main.py --data CIFAR100 --model CIFAR100_ResNEst_ResNet_110 --device cuda:1 --mu 0.01
python3 main.py --data CIFAR100 --model CIFAR100_ResNEst_ResNet_20 --device cuda:1 --mu 0.01
python3 main.py --data CIFAR100 --model CIFAR100_ResNEst_WRN_16_8 --device cuda:1 --mu 0.01
python3 main.py --data CIFAR100 --model CIFAR100_ResNEst_WRN_40_4 --device cuda:1 --mu 0.01
python3 main.py --data CIFAR100 --model CIFAR100_AResNEst_ResNet_110 --device cuda:1
python3 main.py --data CIFAR100 --model CIFAR100_AResNEst_ResNet_20 --device cuda:1
python3 main.py --data CIFAR100 --model CIFAR100_AResNEst_WRN_16_8 --device cuda:1
python3 main.py --data CIFAR100 --model CIFAR100_AResNEst_WRN_40_4 --device cuda:1
