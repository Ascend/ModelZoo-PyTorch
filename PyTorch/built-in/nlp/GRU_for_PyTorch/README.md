一、准数据集：<br>
下载torchtext数据集multi30k。<br>
二、安装依赖包：<br>
1.安装spacy：<br>
`pip3.7 install spacy==2.0.18`

2.安装分词包，使用以下命令或登录对应网址下载分词包<br>
`wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz`

`wget https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-2.0.0/de_core_news_sm-2.0.0.tar.gz`

解压并在分词包目录下，使用以下命令安装：<br>
`python3.7 setup.py install`

三、训练流程：<br>
单卡训练流程：<br>
	1.安装环境<br>
	2.run_1p.sh文件修改字段--npu（单卡训练所使用的device id），为训练配置device_id，比如--npu 0<br>
	3.cd到run_1p.sh文件的目录，执行bash run_1p.sh单卡脚本， 进行单卡训练<br>
	4.run_1p.sh文件可配置内容如下：
	
`        python3.7 gru_1p.py \
            --workers 40 \  #配置模型进程数
            --dist-url 'tcp://127.0.0.1:50000' \
            --world-size 1 \
            --npu 0 \   #使用device 
            --batch-size 512 \ #配置模型batchsize
            --epochs 10 \  #配置模型epochs
            --rank 0 \ 
            --amp \   #使用混合精度加速`
                
多卡训练流程：<br>
	1.安装环境<br>
	2.cd到run_8p.sh文件的目录，执行bash run_8p.sh等多卡脚本， 进行多卡训练<br>	
	3.如需使用0,1,2,3卡，进行4p训练，则修改--device-list '0,1,2,3'<br>
	--workers和--batch-size减少相应倍数，分别修改为workers80，batch-size2048，2p、6p同理
	
`        python3.7 gru_8p.py \
            --addr=$(hostname -I |awk '{print $1}') \ #自动获取主机ip
            --seed 123456 \ #固定随机种子
            --workers 160 \ #配置模型进程数
            --print-freq 1 \ #每*个step打印结果
            --dist-url 'tcp://127.0.0.1:50000' \
            --dist-backend 'hccl' \ 
            --multiprocessing-distributed \ #多p训练
            --world-size 1 \ 
            --batch-size 4096 \  #配置模型batchsize
            --epoch 10 \ #配置模型epoch
            --rank 0 \
            --device-list '0,1,2,3,4,5,6,7' \ #多p训练使用device
            --amp \  #使用混合精度加速`
    	
四、Docker容器训练：

1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如：

    docker import ubuntuarmpytorch.tar pytorch:b020
2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

    ./docker_start.sh pytorch:b020 ./data/multi30k /home/GRU
3.执行步骤一训练流程（环境安装除外）

五、测试结果

训练日志路径：在训练脚本的同目录下result文件夹里，如：

    ./result/training_8p_job_20201121023601