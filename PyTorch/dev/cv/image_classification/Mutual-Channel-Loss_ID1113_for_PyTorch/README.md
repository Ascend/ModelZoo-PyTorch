# The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification

Code release for The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification (TIP 2020)
[DOI](https://doi.org/10.1109/TIP.2020.2973812 "DOI")


## Changelog
- 2020/09/14 update the code: CUB-200-2011_ResNet18.py  Training with ResNet18 (TRAINED FROM SCRATCH).
- 2020/04/19 add the hyper-parameter fine-tune results. 
- 2020/04/18 clean the code for better understanding.

## Dataset
### CUB-200-2011

## Requirements

- python 3.6
- PyTorch 1.2.0
- torchvision

## Training
- Download datasets
- Train: `python CUB-200-2011.py`, the alpha and beta are the hyper-parameters of the  `MC-Loss`
- Description : PyTorch CUB-200-2011 Training with VGG16 (TRAINED FROM SCRATCH).

## Hyper-parameter
Loss = ce_loss + alpha_1 * L_dis + beta_1 * L_div  
![Hyper-parameter_1](https://github.com/dongliangchang/Mutual-Channel-Loss/blob/master/Hyper-parameter_1.jpg)
![Hyper-parameter_2](https://github.com/dongliangchang/Mutual-Channel-Loss/blob/master/Hyper-parameter_2.jpg)
The figure is plot by NNI.



## Other versions
Other unofficial implements can be found in the following:
- Kurumi233: This repo integrate the MC-Loss into a class.  [code](https://github.com/Kurumi233/Mutual-Channel-Loss "code") 
- darcula1993: This repo implement the tf version of the MC-Loss. [code](https://github.com/darcula1993/Mutual-Channel-Loss "code") 


## Citation
If you find this paper useful in your research, please consider citing:
```
@ARTICLE{9005389, 
author={D. {Chang} and Y. {Ding} and J. {Xie} and A. K. {Bhunia} and X. {Li} and Z. {Ma} and M. {Wu} and J. {Guo} and Y. {Song}}, 
journal={IEEE Transactions on Image Processing}, 
title={The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification}, 
year={2020}, volume={29}, number={}, pages={4683-4695}, 
doi={10.1109/TIP.2020.2973812}, 
ISSN={1941-0042}, 
month={},} 
```


## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- changdongliang@bupt.edu.cn
- mazhanyu@bupt.edu.cn


## 网络训练情况

FuncStatus:OK
PerfStatus:NOK
PrecisionStatus:POK