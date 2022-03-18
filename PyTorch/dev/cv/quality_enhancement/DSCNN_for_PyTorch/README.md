环境
----------
    pytorch 1.5
    torchvision 0.5.0
    apex 0.1
    numpy
    Pillow 8.1.0

Host侧训练步骤
----------
1.准备数据集<br>
2.修改run1p.sh，设置训练参数。device_id是用哪张卡，train_data_path是训练网络输入数据，label_data_path是label数据，lr是学习率，model_save_dir是模型保存目录，epoch是训练多少epoch，batch_size是batchsize<br>
3.执行bash run1p.sh开启单p训练<br>


Docker侧训练步骤
----------
    
1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

        docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径(训练和评测数据集都放到该路径下)；模型执行路径；比如：

        ./docker_start.sh pytorch:b020 /train/imagenet /home/CRNN_for_Pytorch
