一、依赖
* Python 3.7.5
* NPU的pytorch训练环境
* requirements.txt下的包
* torch-warmup-lr

二、训练流程：

训练流程：

```
	1.安装python3.7.5的NPU的pytorch训练环境
	2.UCAS-AOD数据集获取：
	    下载地址：https://hyper.ai/datasets/5419
	    下载解压后把数据集改名为UCAS_AOD,存储位置为UCAS_AOD_PATH
	    export UCAS_AOD_PATH=数据集路径
	3.安装requirements.txt
	    安装依赖库
	    apt-get install libgl1 libgeos-dev
	    安装requirements.txt
	    pip3 install -r requirements.txt
	4.安装torch-warmup-lr
	    处理ca证书不通过问题
	    apt-get install ca-certificates
	    git config --global http.sslverify false
	    安装torch-warmup-lr
	    git clone https://github.com/lehduong/torch-warmup-lr.git
	    cd torch-warmup-lr
	    python3.7 setup.py install
	    cd ..
	5.编译hostcpu算子
	    cd utils
	    sh make.sh
	    cd ..
	6.数据预处理
	    bash ./prepare.sh
	7.开始训练
	    训练启动脚本：run_8p.sh run_1p.sh
	    1p训练：bash ./run_1p.sh
	    8p训练：bash ./run_8p.sh
	    第一次训练如果预训练模型下载失败，可以查看下载日志，获取下载地址和存放路径，手动下载后放到对应位置
	8.获得推理结果
	    推理启动脚本：eval.sh
	    bash ./eval.sh
```

三、测试结果
    
训练日志路径：在训练脚本的同目录下weights中

推理结果路径：在训练脚本的同目录下datasets/evaluate/output中

四、备注

不同服务器opencv版本可能不一样。

测试服务器和opencv版本：

Ubuntu aarch64

opencv-python==4.5.4.60