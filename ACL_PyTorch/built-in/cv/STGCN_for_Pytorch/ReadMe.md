## ST_GCN模型推理指导：

### 环境依赖
    pip3 install torch==1.2.0
    pip3 install torchvision==0.4.0
    pip3 install numpy==1.19.2
    pip3 install onnx==1.7.0

### 环境准备
    部署模型代码，安装相关依赖
		下载源码并解压，将st_gcn_export.py、st_gcn_preprocess.py、st_gcn_postprocess.py等文件拷到mmskeleton目录下
		```
		git clone https://github.com/open-mmlab/mmskeleton.git
		```
        
    获取权重文件
		下载权重并解压到权重目录./checkpoints/
		```
		https://download.openmmlab.com/mmskeleton/checkpoints/st_gcn_kinetics-6fa43f73.pth
		```
    
    获取数据集
		从开源链接下载数据集，解压到数据目录./data/kinetics-skeleton/
		目录下存放验证集数据文件val_data.npy，标签文件为val_label.pkl，经过预处理脚本处理过后的数据和标签bin文件，
		将保存到./data/kinetics-skeleton/val_data以及./data/kinetics-skeleton/val_label文件夹。
		```
		https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb
		```     
	
### 数据预处理
    数据集预处理
		```
		python3.7 st_gcn_preprocess.py -data_dir=./data/kinetics-skeleton/val_data.npy -label_dir=./data/kinetics-skeleton/val_label.pkl
		-output_path=./data/kinetics-skeleton/ -batch_size=1 -num_workers=1
		```

    数据集创建
		```
		python3.7 get_info.py bin ./data/kinetics-skeleton/val_data/ kinetics.info 300 18
		```
    
### 离线推理测试
    设置环境变量
		```
		source env.sh
		```    

    转换离线模型
		```
		bash test/pth2om.sh
		```
    
    离线推理测试
		```
		bash test/perf_npu.sh
		```
    
### 精度统计  
		```
		bash test/eval_acc.sh
		```