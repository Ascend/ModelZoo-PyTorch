# Multi-Task Network for Vehicle Re-Identification
实现了Multi-Task Network在VeRi数据集上的训练。
- 参考实现
```
url=https://github.com/NVlabs/PAMTRI
branch=master
commit_id=25564bbebd3ccf11d853a345522e2d8c221b275d

```
# Multi-Task Network Detail
- 增加了混合精度训练
- 增加了多卡分布式训练	

# Requirements
- CANN 5.0.2及对应版本的PyTorch
- pip install -r requirements.txt
  注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0
- 下载[VeRi数据集](https://pan.baidu.com/s/1gYBNQI0_MZLB0ANW8qnYGw)，密码：zqk4
- 下载[数据集的label文件](https://github.com/NVlabs/PAMTRI.git)，在PAMTRI/MultiTaskNet/data/veri/下获取label.csv文件。
  <br>将数据集解压在./data/veri/目录下，应包含image_train、image_test、image_query三个数据集文件以及相应的.csv文件。
- 下载[模型文件](https://github.com/NVlabs/PAMTRI.git)，在PAMTRI/MultiTaskNet/model/下获取模型文件，并将模型文件放入./model/densenet121-xent-htri-veri-multitask/目录下
# 自检报告

```
#1p train full
# 是否正确输出了精度log文件，是否正确保存了模型文件
bash ./test/train_full_1p.sh --data_path=real_data_path
# 备注： 目标精度69.46；验收精度69.27；无输出日志，运行报错，报错日志xx 等

#1p train perf
# 是否正确输出了性能log文件
bash ./test/train_performance_1p.sh --data_path=real_data_path
# 验收结果： OK / Failed
# 备注： 目标性能200FPS；验收测试性能89.3FPS；无输出日志，运行报错，报错日志xx 等

#8p train full
# 是否正确输出了精度log文件，是否正确保存了模型文件
bash ./test/train_full_8p.sh --data_path=real_data_path
# 备注： 目标精度54.31；验收精度54.34；无输出日志，运行报错，报错日志xx 等

#8p train perf
# 是否正确输出了性能log文件
bash ./test/train_performance_8p.sh --data_path=real_data_path
# 验收结果： OK / Failed
# 备注： 目标性能1302FPS；验收测试性能503FPS；无输出日志，运行报错，报错日志xx 等

#eval
# 是否正确进行模型评估文件
bash ./test/eval.sh --data_path=real_data_path --pth_path=real_pre_train_model_path
# 验收结果： OK / Failed
# 备注： 无输出日志，运行报错，报错日志xx 等

#To ONNX
# 是否正确进行pth模型文件转onnx模型文件
python3 PAMTRI_pth2onnx.py --load-weights ./real_model_path.pth --output_path ./PAMTRI.onnx
# 验收结果： OK / Failed
# 备注： 无输出日志，运行报错，报错日志xx 等

```
## Multi-Task Net training result
Device | mAP | FPS | Epochs | AMP_Type
---|---|---|---|---|
GPU-1P | 69.36 | 200 | 120 | O2
GPU-8P | 54.31 | 1302 | 120 | O2 
NPU-1P | 69.27 | 89.3 | 120 | O2 
NPU-8P | 54.34 | 503 | 120 | O2  