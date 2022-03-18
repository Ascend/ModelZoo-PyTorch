#!/bin/bash
python3 main.py --data CIFAR10 --model CIFAR10_Standard_ResNet_110 --device cuda:0
python3 main.py --data CIFAR10 --model CIFAR10_Standard_ResNet_20 --device cuda:0
python3 main.py --data CIFAR10 --model CIFAR10_Standard_WRN_16_8 --device cuda:0
python3 main.py --data CIFAR10 --model CIFAR10_Standard_WRN_40_4 --device cuda:0
python3 main.py --data CIFAR10 --model CIFAR10_BNResNEst_ResNet_110 --device cuda:0
python3 main.py --data CIFAR10 --model CIFAR10_BNResNEst_ResNet_20 --device cuda:0
python3 main.py --data CIFAR10 --model CIFAR10_BNResNEst_WRN_16_8 --device cuda:0
python3 main.py --data CIFAR10 --model CIFAR10_BNResNEst_WRN_40_4 --device cuda:0
python3 main.py --data CIFAR100 --model CIFAR100_Standard_ResNet_110 --device cuda:0
python3 main.py --data CIFAR100 --model CIFAR100_Standard_ResNet_20 --device cuda:0
python3 main.py --data CIFAR100 --model CIFAR100_Standard_WRN_16_8 --device cuda:0
python3 main.py --data CIFAR100 --model CIFAR100_Standard_WRN_40_4 --device cuda:0
python3 main.py --data CIFAR100 --model CIFAR100_BNResNEst_ResNet_110 --device cuda:0
python3 main.py --data CIFAR100 --model CIFAR100_BNResNEst_ResNet_20 --device cuda:0
python3 main.py --data CIFAR100 --model CIFAR100_BNResNEst_WRN_16_8 --device cuda:0
python3 main.py --data CIFAR100 --model CIFAR100_BNResNEst_WRN_40_4 --device cuda:0
