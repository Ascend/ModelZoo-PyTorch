# ICNet 

## Description
- This repo contains ICNet implemented by PyTorch, based on [paper](https://arxiv.org/abs/1704.08545) by Hengshuang Zhao, and et. al(ECCV'18).  
- Training and evaluation are done on the [Cityscapes dataset](https://www.cityscapes-dataset.com/) by default.
- Base version of the model from [the paper author's code on Github](https://github.com/liminn/ICNet-pytorch).  

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
            pip3.7 install --upgrade torchvision 
    - 其他依赖包 pip install -r requirements.txt  
      
            torchsummary==1.5.1
            Pillow==9.1.0  
            PyYAML==5.1.2  
            requests==2.22.0  
            tqdm==4.19.9  
            numpy  
            sympy  
            decorator  
      注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0

##Training

- 1p training 1p 
    - bash ./test/train_full_1p.sh --data_path=xxx # training accuracy

    - bash ./test/train_performance_1p.sh --data_path=xxx # training performance

- 8p training 8p
    - bash ./test/train_full_8p.sh --data_path=xxx # training accuracy
    
    - bash ./test/train_performance_8p.sh --data_path=xxx # training performance

- eval default 8p， should support 1p
    - bash ./test/train_eval_8p.sh --data_path=xxx

- Traing log
    - test/output/devie_id/train_${device_id}.log # training detail log
    
    - test/output/devie_id/ResNet101_${device_id}_bs_8p_perf.log # 8p training performance result
    
    - test/output/devie_id/ResNet101_${device_id}_bs_8p_acc.log # 8p training accuracy result    

    
- 在线推理流程  
    - 进入代码目录
    - 将configs/icnet_eval.yaml中的参数"cityscapes_root"、"ckpt_dir"和"ckpt_path"修改为实际路径    
    - 执行bash scripts/eval.sh开始推理，代码默认单卡  
 
   
- demo测试
    - 进入代码目录
    - 将configs/icnet_eval.yaml中的参数"cityscapes_root"、"ckpt_dir"和"ckpt_path"修改为实际路径    
    - 执行python3.7.5 demo.py进行demo测试
    

- pth转onnx
    - 进入代码目录
    - 修改pth_file参数为实际转换的pth文件名
    - 执行python3.7.5 pth2onnx.py进行onnx转换，转换成功后，目录下生成ICNet.onnx文件  
    

- 测试指令合集  
    - 1p train  
        bash scripts/train_1p.sh
    - 8p train  
        bash scripts/train_8p.sh
    - eval default 1p  
        bash scripts/eval.sh
    - online inference demo   
        python3.7.5 demo.py
    - To ONNX  
        python3.7.5 pth2onnx.py

## ICNet training result

| 名称         | mIoU      | FPS       |
| :---------:  | :------: | :------:  |
| 官方源码     | 68.4     | -         |  
| GPU-1p      | -        | 9.3       | 
| GPU-8p      | 68.3     | 83        | 
| NPU-1p      | -        | 9.3       | 
| NPU-8p      | 68.1     | 83        | 

- 说明  
    
    - 官网中，模型指标标注的是mIoU 71%，但是实测结果只有68.4%，无法达到官网指标。  
      官网参考Base version of the model from [the paper author's code on Github](https://github.com/liminn/ICNet-pytorch).
    - GPU-8p精度对齐过程，通过调整超参，最终精度和最优精度分别如下：  
    
    | batch_size   | lr       | loss_scale | opt_level  | final mIoU  | best mIoU   |    
    | :---------:  | :------: | :------:   | :--------: | :---------: | :---------: |    
    | 16           | 0.06     | 128        |  O1        | 68.1        |  68.7       |  
    | 16           | 0.08     | 128        |  O1        | 68.3        |  68.6       |  
    | 16           | 0.1      | 128        |  O1        | 67.7        |  68.6       |  
    | 16           | 0.08     | 128        |  O2        | 67.7        |  68.0       |
      
    - NPU-8p精度对齐过程，通过调整超参，最终精度和最优精度分别如下：
    
    | batch_size   | lr       | loss_scale | opt_level  | final mIoU  | best mIoU   |    
    | :---------:  | :------: | :------:   | :--------: | :---------: | :---------: |    
    | 16           | 0.08     | 32         |  O1        | 67.6        |  68.0       |  
    | 16           | 0.1      | 32         |  O1        | 68.1        |  68.3       |  
    | 16           | 0.11     | 32         |  O1        | 67.9        |  68.4       |  
    | 16           | 0.12     | 32         |  O1        | 67.0        |  67.5       |
    | 16           | 0.1      | 32         |  O2        | 66.5        |  67.0       |
    | 16           | 0.1      | 128        |  O1        | 67.0        |  67.4       |
