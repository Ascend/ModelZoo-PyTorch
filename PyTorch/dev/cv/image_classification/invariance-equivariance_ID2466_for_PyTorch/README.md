# Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning

Implementation of [Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning](https://arxiv.org/abs/2103.01315). The paper reports state-of-the-art results on ***five popular few-shot learning benchmarks: CIFAR-FS, FC100, miniImageNet, tieredImageNet, and Meta-Dataset.***

In many real-world problems, collecting a large number of labeled samples is infeasible. Few-shot learning (FSL) is the dominant approach to address this issue, where the objective is to quickly adapt to novel categories in presence of a limited number of samples. FSL tasks have been predominantly solved by leveraging the ideas from gradient-based meta-learning and metric learning approaches. However, recent works have demonstrated the significance of powerful feature representations with a simple embedding network that can outperform existing sophisticated FSL algorithms. In this work, we build on this insight and propose a novel training mechanism that simultaneously enforces equivariance and invariance to a general set of geometric transformations. Equivariance or invariance has been employed standalone in the previous works; however, to the best of our knowledge, they have not been used jointly. Simultaneous optimization for both of these contrasting objectives allows the model to jointly learn features that are not only independent of the input transformation but also the features that encode the structure of geometric transformations. These complementary sets of features help generalize well to novel classes with only a few data samples. We achieve additional improvements by incorporating a novel self-supervised distillation objective. Our extensive experimentation shows that even without knowledge distillation our proposed method can outperform current state-of-the-art FSL methods on five popular benchmark datasets.

This repository is implemented using PyTorch and it includes code for running the few-shot learning experiments on CIFAR-FS, FC-100, miniImageNet and tieredImageNet datasets.

<p align="center">
  <img src="/figures/conceptual-1.png" width="500">
</p>
<p>
  <em>Approach Overview: Shapes represent different transformations and colors represent different classes. While the invariant features provide better discrimination, the equivariant features help us learn the internal structure of the data manifold. These complimentary representations help us generalize better to new tasks with only a few training samples. By jointly leveraging the strengths of equivariant and invariant features, our method achieves significant improvement over baseline (bottom row).</em>
</p>

<p align="center">
  <img src="/figures/training.png" width="800">
</p>
<p>
  <em>Network Architecture during Training: A series of transformed inputs (transformed by applying transformations T1...TM) are
provided to a shared feature extractor fΘ. The resulting embedding is forwarded to three parallel heads fΨ, fΦ and fΩ that focus on
learning equivariant features, discriminative class boundaries, and invariant features, respectively. The resulting output representations are
distilled from an old copy of the model (teacher model on the right) across multiple-heads to further improve the encoded representations.
Notably, a dedicated memory bank of negative samples helps stabilize our invariant contrastive learning.</em>
</p>

## Presentation
[![Presentation: Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning](https://yt-embed.herokuapp.com/embed?v=NfE1CXqzE8s)](https://www.youtube.com/watch?v=NfE1CXqzE8s&start=1100 "Presentation: Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning")


## Dependencies
This code requires the following:
* Python >= 3.6
* numpy==1.16.2
* Pillow==5.4.1
* scikit-learn==0.21.1
* scipy==1.2.1
* torch==1.3.0
* torchvision==0.8.1
* tqdm==4.36.1
* tensorboardx==1.7
* tensorboard==1.13.1
* matplotlib==3.2.1
* mkl==2019.0
* wandb==0.8.36

run `pip3 install -r requirements.txt` to install all the dependencies. 

## Download Data
The data we used here is preprocessed by the repo of [MetaOptNet](https://github.com/kjunelee/MetaOptNet), Please find the renamed versions of the files in below link by [RFS](https://github.com/WangYueFt/rfs).

[[DropBox Data Packages Link]](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0)


## Training

### Invariance-Equivariance
To perform the experiment with invariance and equivariance, run:

```shell
# For CIFAR-FS
python3 train.py --model resnet12 --model_path save --dataset CIFAR-FS --data_root /path_to_data_folder(should contain subfolder CIFAR-FS) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --epochs 65 --lr_decay_epochs 60 --gamma 1.0 --contrast_temp 1.0 --mvavg_rate 0.99 --memfeature_size 64 --batch_size 64 --tags CIFAR-FS,INV_EQ

# For FC100
python3 train.py --model resnet12 --model_path save --dataset FC100 --data_root /path_to_data_folder(should contain subfolder FC100) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --epochs 65 --lr_decay_epochs 60 --gamma 1.0 --contrast_temp 1.0 --mvavg_rate 0.99 --memfeature_size 64 --batch_size 64 --tags FC100,INV_EQ

# For miniImageNet (require multiple GPUs)
python3 train.py --model resnet12 --model_path save --dataset miniImageNet --data_root /path_to_data_folder(should contain subfolder miniImageNet) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --epochs 65 --lr_decay_epochs 60 --gamma 1.0 --contrast_temp 1.0 --mvavg_rate 0.99 --memfeature_size 64 --batch_size 32 --tags miniImageNet,INV_EQ

# For tieredImageNet (require multiple GPUs)
python3 train.py --model resnet12 --model_path save --dataset tieredImageNet --data_root /path_to_data_folder(should contain subfolder tieredImageNet) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --epochs 60 --lr_decay_epochs 30,40,50 --gamma 1.0 --contrast_temp 1.0 --mvavg_rate 0.99 --memfeature_size 64 --batch_size 32 --tags tieredImageNet,INV_EQ
```

WANDB will create unique names for each runs, and save the model names accordingly. Use this name for the teacher in the next experiment.


### Invariance-Equivariance-Distill
To perform the distillation experiment, run: 

```shell
# For CIFAR-FS
python3 train_distillation.py --model_s resnet12 --model_t resnet12 --path_t /path_to_teacher_model --model_path save --dataset CIFAR-FS --data_root /path_to_data_folder(should contain subfolder CIFAR-FS) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --epochs 65 --lr_decay_epochs 60 --gamma 1.0 --w_ce 1.0 --w_div 1.0 --kd_T 4 --contrast_temp 1.0 --mvavg_rate 0.99 --memfeature_size 64 --batch_size 64 --tags CIFAR-FS,INV_EQ_DISTILL

# For FC100
python3 train_distillation.py --model_s resnet12 --model_t resnet12 --path_t /path_to_teacher_model --model_path save --dataset FC100 --data_root /path_to_data_folder(should contain subfolder FC100) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --epochs 65 --lr_decay_epochs 60 --gamma 1.0 --w_ce 1.0 --w_div 1.0 --kd_T 4 --contrast_temp 1.0 --mvavg_rate 0.99 --memfeature_size 64 --batch_size 64 --tags FC100,INV_EQ_DISTILL

# For miniImageNet (require multiple GPUs)
python3 train_distillation.py --model_s resnet12 --model_t resnet12 --path_t /path_to_teacher_model --model_path save --dataset miniImageNet --data_root /path_to_data_folder(should contain subfolder miniImageNet) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --epochs 65 --lr_decay_epochs 60 --gamma 1.0 --w_ce 1.0 --w_div 1.0 --kd_T 4 --contrast_temp 1.0 --mvavg_rate 0.99 --memfeature_size 64 --batch_size 32 --tags miniImageNet,INV_EQ_DISTILL

# For tieredImageNet (require multiple GPUs)
python3 train_distillation.py --model_s resnet12 --model_t resnet12 --path_t /path_to_teacher_model --model_path save --dataset tieredImageNet --data_root /path_to_data_folder(should contain subfolder tieredImageNet) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --epochs 60 --lr_decay_epochs 30,40,50 --gamma 1.0 --w_ce 1.0 --w_div 1.0 --kd_T 4 --contrast_temp 1.0 --mvavg_rate 0.99 --memfeature_size 64 --batch_size 32 --tags tieredImageNet,INV_EQ_DISTILL
```


### Evaluation
Each of the training run will provide the final evaluation scores. However, for separate evaluation, run:
```shell
# For CIFAR-FS
python3 eval_fewshot.py --model resnet12 --dataset CIFAR-FS --data_root /path_to_data_folder(should contain subfolder CIFAR-FS) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --model_path /path_to_model

# For FC100
python3 eval_fewshot.py --model resnet12 --dataset FC100 --data_root /path_to_data_folder(should contain subfolder FC100) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --model_path /path_to_model

# For miniImageNet
python3 eval_fewshot.py --model resnet12 --dataset miniImageNet --data_root /path_to_data_folder(should contain subfolder miniImageNet) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --model_path /path_to_model

# For tieredImageNet
python3 eval_fewshot.py --model resnet12 --dataset tieredImageNet --data_root /path_to_data_folder(should contain subfolder tieredImageNet) --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --model_path /path_to_model
```

## We Credit
Thanks to https://github.com/WangYueFt/rfs, for the preliminary implementations.

## Citation
```
@InProceedings{Rizve_2021_CVPR,
    author    = {Rizve, Mamshad Nayeem and Khan, Salman and Khan, Fahad Shahbaz and Shah, Mubarak},
    title     = {Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {10836-10846}
}
```
