# X3D-S

Implements training of X3D-S on the Kinetics-400 dataset

## Detail

Most of codes are modified according to [here](https://gitee.com/ascend/modelzoo/wikis/Pytorch%E8%AE%AD%E7%BB%83%E6%8C%87%E5%AF%BC?sort_id=4208869#21-%E8%BF%81%E7%A7%BB%E6%B5%81%E7%A8%8B%E8%AF%B4%E6%98%8E)

There are some special modification of [source repository](https://github.com/facebookresearch/SlowFast) :

##### NPU & GPU

- Add some customized yaml configuration items, such as APEX.ENABLE、DIST_BACKEND...
- Ascend-Pytorch-1.5 is not supported `torch.nn.init.trunc_normal` , using `torch.nn.init.normal_`  instead
- Adjusted the order of dependency import to prevent some unknown bugs (`scikit-learn`)

##### NPU

- Group conv3D of Ascend-Pytorch is not supported, so we canceled all group operations in the model
- Remove some irrelevant codes to prevent some unknown bugs (`Segmentation fault (core dumped)`)


## Requirements

##### Base Environment

- Python == 3.7.5
- GCC >= 4.9

##### Python Environment

1. Installing  these error-prone dependencies first:

- PyTorch （raw==1.5 or ascend）
  - Ascend-Pytorch Version after August 24 be installed 
- torchvision == 0.6.0
  - If on Centos arm, please build the source code from [here](https://gitee.com/azureology/torchvision/tree/v0.6.0/) 
- PyAV
  - If the installation fails on Centos arm, following this [issue](https://gitee.com/ascend/modelzoo/issues/I48AP3?from=project-issue)
- Detectron2
  - According to the CUDA version and Pytorch version, build from [source code](https://github.com/facebookresearch/detectron2)

2. Then, you can use `pip3 install -r requirements.txt`   to install some simple dependencies 



3. Building source code

```shell
cd X3D  # into source code root

# Switch to your prepared environment

python3 setup.py build develop  # build slowfast and install the remaining dependencies 
```



##### Modify  Ascend-tookit

```shell
cd /usr/local
find / -name fractal_z_3d_2_ncdhw.py
vi path_to_fractal_z_3d_2_ncdhw.py

located method: 
    1. def fractal_z_3d_2_ncdhw(src, dst, src_format, dst_format,kernel_name="fractal_z_3d_2_ncdhw")
	2. modify it according this picture:
		2.1. remove `if list(dst_shape) in ....`
		2.2. Align the next two lines like this
```

![image-20210909203603647](meta\bug-opt.png)



##### Dataset

- Download the Kinetics-400 dataset from [here](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/dataset/k400.md)

  1. unzip the all packages and merge all folders

  2. we get two sets , train set (used to train) and val set (used to test). And each of both has 400 folders

     ```markdown
     # format of data folder
     
     |-data
     	 |-train
     	 	|- video type 1
     	 		|- video 1
     	 		|- video 2
     	 		...
     	 	|- video type 2
     	 	...
     	 	|- video type 400
     	 |-val
     	 	|- video type 1
     	 	|- video type 2
     	 	...
     	 	|- video type 400
     ```
  
     
  
  3. build train.csv, val.csv, test.csv, and put them in the same folder
  
     ```markdown
     # format of data path folder
     |- data_path
     	|- train.csv
     	|- val.csv
     	|- test.csv
     ```
  
     train.csv consists of train set
  
     val.csv is same as test.csv, and consists of test set
  
     ```markdown
     # format of data path csv is:
     
     path_to_video_1 label_1
     path_to_video_2 label_2
     ...
     path_to_video_N label_N
     ```
  
  4. check if the all videos are lossless according to the scripts provided by project [mmaction2](https://github.com/open-mmlab/mmaction2) . Here, we  provide the [list](mytest\Vinput\error_video) of  corrupted videos that have been checked out
5. remove the those corrupted videos from the three csv

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

> Note：the `real_data_path` is path of csv folder mentioned above

```bash
# training 1p (300 epoch)
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 8p (300 epoch)
bash ./test/train_full_8p.sh --data_path=real_data_path

# training performance 1p (1 epoch)
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training performance 8p (3 epoch)
bash ./test/train_performance_8p.sh --data_path=real_data_path

# testing 8p
bash test/train_eval_8p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path

# train_finetune_1p.sh
bash test/train_finetune_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path --num_classes=default_400
```

> Log path: ./stdout.log



## Training Result

> Due to the calculation cast a long time, we choose to run the full data, and the NPU-ACC is aligned with the GPU-ACC  (as many epochs as possible).


|     Device      |   FPS   | Top1-ACC 10-view | Batch Size | Epochs |   AMP    |
| :-------------: | :-----: | :--------------: | :--------: | :----: | :------: |
|     1P-GPU      |  10.39  |      6.67%       |     96     | 1/300  | O2-128.0 |
|     1P-NPU      |  5.38   |      6.18%       |     96     | 1/300  | O2-128.0 |
|  1P-NPU-白名单  |  5.35   |      6.36%       |     96     | 1/300  | O2-128.0 |
|     8P-GPU      | 1137.49 |      37.56%      |    256     | 30/300 | O2-128.0 |
|     8P-NPU      | 529.24  |      39.67%      |    256     | 30/300 | O2-128.0 |
| 8P-NPU-fusedSGD | 510.66  |      5.80%       |    256     | 2/300  | O2-128.0 |



- Testing result: Top1-ACC of 8P-NPU and 8P-GPU training (30 epochs)

![img](meta\8P-GPU & 8P-NPU.png)

## Performance Optimization

According to the above, it can be concluded that the accuracy(Top1-ACC 10-view) of 8P-GPU and 8P-NPU is  little different. But performance(FPS) of 8P-NPU is 50% of 8P-GPU's.

So we made the following analysis and improvement:

- find the dynamic operators following [here](https://gitee.com/wangjiangben_hw/ascend-pytorch-crowdintelligence-doc/blob/master/pytorch-train-guide/%E6%A8%A1%E5%9E%8B%E7%AE%97%E5%AD%90%E6%8F%90%E5%8F%96%E6%8C%87%E5%8D%97.md), but the operators is very basic, and we can not identify them from our big model.

![img](meta\dynamic_ops.png)

- check the profile of NPU through chrome tracing

  <img src="meta\profile-1.png" alt="image-20210913190836215" style="zoom: 50%;" />

- In order to improve the low perfomance of Transpose, we first generate the `cann profiling` following [here](https://gitee.com/wangjiangben_hw/ascend-pytorch-crowdintelligence-doc/blob/master/pytorch-train-guide/CANNProfiling%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC%E4%B9%A6.md), then we extract the two operators, TransposeD and TransData.
  - if TransposeD  `Consuming time > 10s`, add its Input Shapes to White List (/usr/local/Ascend/ascend-toolkit/5.0.2/x86_64-linux/opp/op_impl/built-in/ai_core/tbe/impl/dynamic/transpose.py）
  - if TransData `Consuming time > 10s & Input Formats == 'NCHW' & Output Formats == 'NC1HWC0'`, add its Input Shapes to White List (/usr/local/Ascend/ascend-toolkit/5.0.2/x86_64-linux/opp/op_impl/built-in/ai_core/tbe/impl/four_2_five.py）
  - if TransData `Consuming time > 10s & Input Formats == 'NC1HWC0' & Output Formats == 'NCHW'`, add its Input Shapes to White List (/usr/local/Ascend/ascend-toolkit/5.0.2/x86_64-linux/opp/op_impl/built-in/ai_core/tbe/impl/five_2_four.py）

**After Optimization**

![image-20210918103240921](meta\profile-2.png)

## ELSE

Iessues and PRs about this project

- invalid gradient https://gitee.com/ascend/modelzoo/issues/I452ZB  https://gitee.com/ascend/pytorch-develop/pulls/2438
- optimizer error https://gitee.com/ascend/pytorch-develop/pulls/2438
- pyav install on CentOS arm  https://gitee.com/ascend/modelzoo/issues/I48AP3
- scikit-learn cannot allocate memory in static TLS https://gitee.com/ascend/modelzoo/issues/I48QNY


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md