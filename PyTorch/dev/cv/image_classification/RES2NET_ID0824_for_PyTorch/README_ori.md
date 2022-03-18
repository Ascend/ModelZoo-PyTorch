# Training code of Res2Net on ImageNet

The ImageNet pretrained models is available in [Res2Net-PretrainedModels](https://github.com/Res2Net/Res2Net-PretrainedModels).

This repo is the training code of Res2Net on ImageNet. All results listed in the paper are trained with this script.
The training script of Res2Net is based on the older version of pytorch example.
Since the pytorch example been have updated to a new version, to ensure the community to reproduce our results in the paper,
we now open source our ImageNet training code.
**Note that for fair comparision, we followed the common settings and didn't use any other data augmentation tricks such as mixup and warm-up in our paper.**
You can use these tricks if you want to have a better performance.

If you still have the difficulty to reproduce our results, please e-mail me via shgao[at].live.com

Note: Due to our contract with the company, this code can only be used for non-commercial purposes.

More training codes of other tasks based on Res2Net can be found on https://mmcheng.net/res2net and https://github.com/Res2Net


## 网络训练状况：

FuncStatus:OK(流程通过)
PerfStatus:NOK(小于0.5倍GPU)
PrecisionStatus:POK(Loss拟合但精度未实施)