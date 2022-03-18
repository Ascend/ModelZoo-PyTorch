## Requirements
```angular2html
pytorch==1.5
apex
pandas
opencv-python
```

## 下载数据集
```angular2html
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

将3个数据集放在目录 VOC0712下，目录结构为：
    VOC0712
    |
    |———————— VOC2007_trainval
    |         |——————Annotations
    |         |——————ImageSets
    |         |——————JPEGImages
    |         |——————SegmentationClass
    |         |——————SegmentationObject
    |———————— VOC2012_trainval
    |         |——————Annotations
    |         |——————ImageSets
    |         |——————JPEGImages
    |         |——————SegmentationClass
    |         |——————SegmentationObject
    |———————— VOC2007_test
              |——————Annotations
              |——————ImageSets
              |——————JPEGImages
              |——————SegmentationClass
              |——————SegmentationObject
```
## 下载预训练模型到 models 目录下
```
wget -P models https://storage.googleapis.com/models-hao/mb2-imagenet-71_8.pth
```
## 训练
```angular2html
# 1p train perf
# 是否正确输出了性能log文件
bash test/train_performance_1p.sh --data_path xxx

# 1p train full
# 是否正确输出了性能精度log文件，是否正确保存了模型文件
bash test/train_full_1p.sh --data_path xxx 

# 8p train perf
# 是否正确输出了性能log文件
bash test/train_performance_8p.sh --data_path xxx

# 8p train full
# 是否正确输出了性能精度log文件，是否正确保存了模型文件
bash test/train_full_8p.sh --data_path xxx 

# finetuning
# 是否正确执行迁移学习
bash test/train_finetune_1p.sh --data_path xxx 

# online inference demo 
# 是否正确输出预测结果，请确保输入固定tensor多次运行的输出结果一致
python3.7.5 demo.py
```
### 一些参数说明
```angular2html
--data_path             数据集路径
--base_net              预训练模型存放路径
--num_epochs            训练epoch
--validation_epochs     验证epoch
--checkpoint_folder     模型保存路径
--eval_dir              模型验证时产生文件的存放路径
--device                使用的设备，npu或gpu
--gpu                   设备卡号，单卡时使用
--device_list           默认为 '0,1,2,3,4,5,6,7'，多卡时使用
```
## evaluate
```angular2html
bash scripts/eval.sh
```
### 一些参数说明
```angular2html
--dataset               测试数据集
--eval_dir              模型验证时产生文件的存放路径
--lable_file            类别文件，训练时会在模型保存文件夹生成
```
