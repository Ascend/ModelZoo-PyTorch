# CSWin-Transformer

- 参考实现：
```
url=https://gitee.com/rainyuniverse/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/classification/CSWin-Transformer 
branch=master
commit_id=2cec9d71ad910f441899a472331a61fc8f82ef36
```

# Requirements

- install requirement

  ```
  pip install bcolz mxnet tensorboardX matplotlib easydict opencv-python einops
  pip install scikit-image imgaug PyTurboJPEG
  pip install scikit-learn
  pip install termcolor prettytable
  ```
  
  安装修改过的`timm`库，
  
  ```shell
  # 卸载已安装的timm库
  pip3 uninstall timm
  
  # 安装修改过的timm库
  cd pytorch-image-models-0.3.4
  python3 setup.py install
  ```
  
  torch 和 apex要固定为ascend20220315版本，之后的版本会变慢。
  
  另外还需要在项目目录下新建`dataset`文件夹，并在`dataset`路径下添加文件，文件下载地址：[CSWin-Transformer/dataset at main · microsoft/CSWin-Transformer (github.com)](https://github.com/microsoft/CSWin-Transformer/tree/main/dataset)

# 精度性能

|  名称  | 精度  | 性能 |
| :----: | :---: | :--: |
| GPU-1p |   -   | 230  |
| GPU-8p | 82.3  | 1700 |
| NPU-1p |   -   | 276  |
| NPU-8p | 82.45 | 2234 |

# 自验报告
```shell    
# 1p train perf
# 是否正确输出了性能log文件
bash test/train_performance_1p.sh --data_path=real_data_path
# 验收结果： OK
    
# 8p train perf
# 是否正确输出了性能log文件
bash test/train_performance_8p.sh --data_path=real_data_path
# 验收结果： OK 

# 1p train full
# 是否正确输出了性能精度log文件，是否正确保存了模型文件
bash test/train_full_1p.sh --data_path=real_data_path
# 验收结果： OK 

# 8p train full
# 是否正确输出了性能精度log文件，是否正确保存了模型文件
bash test/train_full_8p.sh --data_path=real_data_path
# 验收结果： OK 

# 8p eval
# 是否正确输出了性能精度log文件
bash test/train_eval_8p.sh --data_path=real_data_path 
# 验收结果： OK 
```
