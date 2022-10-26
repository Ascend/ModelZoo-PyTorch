一、安装依赖包：
1.安装spacy：
`pip3.7 install spacy==2.0.18`

2.安装分词包，使用以下命令或登录对应网址下载分词包
`wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz`

`wget https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-2.0.0/de_core_news_sm-2.0.0.tar.gz`

解压并在分词包目录下，使用以下命令安装：
`python3.7 setup.py install`


二、准数据集：
下载torchtext数据集multi30k:

三、训练流程：
单卡训练流程：
	1.安装环境
	2.执行脚本 bash ./test/train_performance_1p.sh --data_path=数据集路径            [ 数据集写到multi30k上一层，即不包括multi30k ]
                
多卡训练流程：
	1.安装环境
	2.执行脚本 bash ./test/train_full_8p.sh --data_path=数据集路径       [ 数据集写到multi30k上一层，即不包括multi30k ]                    
	[ 如需使用0,1,2,3卡，进行4p训练，则修改--device-list '0,1,2,3'
	--workers和--batch-size减少相应倍数，分别修改为workers80，batch-size2048，2p、6p同理 ]
    	
四、Docker容器训练：

1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如：

    docker import ubuntuarmpytorch.tar pytorch:b020
2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

    ./docker_start.sh pytorch:b020 ./data/multi30k /home/GRU
3.执行步骤一训练流程（环境安装除外）

五、测试结果

训练日志路径：在训练脚本的同目录下result文件夹里，如：

    ./test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log