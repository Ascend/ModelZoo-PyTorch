# HRNet

## Description
- This implements training of hrnet on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).
- Base version of the model from [the author's code on Github](https://github.com/HRNet/HRNet-Image-Classification).  

## Requirements
- 环境搭建  
    说明：若使用搭建完成的环境，可跳过此步骤。一般情况下，华为默认会提供搭建完成的环境。  
- run包安装  
    - 安装前准备  
        当前适配Ascend后的PyTorch，仅支持在真实Ascend NPU系列推理和训练设备上的运行。  
        在Ascend NPU系列设备上部署开发环境，请参考[《CANN V100R020C10 软件安装指南》](https://support.huawei.com/enterprise/zh/doc/EDOC1100164870/59fb2d06)
        的“获取软件包”和”安装开发环境“章节，完成安装前准备。
    - 完成驱动和固件的安装
        请参考[《CANN V100R020C10 软件安装指南》](https://support.huawei.com/enterprise/zh/doc/EDOC1100164870/59fb2d06)  
      的”安装昇腾芯片驱动和固件“ ->“安装开发套件和框架插件包”章节，完成安装。  
      
            说明:
            安装驱动和固件需要root用户，若使用默认路径安装： 
            ps. 如升级run包，建议 ./A800-9010-NPU_Driver-x.x.x-X86_64- Ubuntu18.04.run --uninstall;rm -rf /usr/local/Ascend;reboot
            安装驱动：./A800-9010-NPU_Driver-x.x.x-X86_64- Ubuntu18.04.run --full --quiet  
            安装固件：./A800-9010-NPU_Firmware-<version>.run --full --quiet  
            安装CANN包：./Ascend-Toolkit-{version}-xxx-{gcc_version}.run --install --quiet
- whl包安装  
  Python 3.6 or later
    - 获取安装Ascend版本的Pytorch和Apex [downlink](https://www.hiascend.com/software/ai-frameworks)  
      
            pip3.7 install --upgrade torch-*.whl  
            pip3.7 install --upgrade apex-*.whl   
            pip3.7 install --upgrade torchvision==0.6.0 # 对应当前torch==1.5.0的版本，arm请使用0.2.2  
    - 其他依赖包 pip install -r requirements.txt  
      
            EasyDict==1.7
            opencv-python==3.4.1.15
            shapely==1.6.4
            Cython
            scipy
            pandas
            pyyaml
            json_tricks
            scikit-image
            yacs>=0.1.5
            tensorboardX>=1.6  

##Training

- training 1p 
    - bash ./test/train_full_1p.sh --data_path=xxx # training accuracy

    - bash ./test/train_performance_1p.sh --data_path=xxx # training performance

- training 8p
    - bash ./test/train_full_8p.sh --data_path=xxx --device_id=xxx # training accuracy  
    
    - bash ./test/train_performance_8p.sh --data_path=xxx  --device_id=xxx  # training performance

- eval default 8p， should support 1p
    - bash ./test/train_eval_8p.sh --data_path=xxx   --device_id=xxx

- Traing log
    - test/output/devie_id/train_${device_id}.log # training detail log
    
    - test/output/devie_id/HRNe_ID1780${device_id}_bs_8p_perf.log # 8p training performance result
    
    - test/output/devie_id/HRNe_ID1780${device_id}_bs_8p_acc.log # 8p training accuracy result    

```bash

# online inference demo 
python3 demo.py

# pth转换onnx
python3 pthtar2onnx.py

```

- 模型训练结果如下：

| 名称         | Acc1      | FPS       |
| :---------:  | :------: | :------:  |
| 官方源码     | 76.8     | -         |
| GPU-1p      | -        | 26.2       | 
| GPU-8p      | 76.8     | 210        | 
| NPU-1p      | -        | 33.3       | 
| NPU-8p      | 76.4     | 218.4      | 

