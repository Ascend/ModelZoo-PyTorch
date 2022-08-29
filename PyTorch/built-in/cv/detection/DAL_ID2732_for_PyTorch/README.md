一、依赖
* Python 3.7.5
* NPU的pytorch训练环境
* requirements.txt下的包
* torch-warmup-lr

注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision
    建议：Pillow版本是9.1.0  torchvision版本是0.6.0

二、训练流程：

训练流程：

```
	1.安装python3.7.5的NPU的pytorch训练环境
	2.UCAS-AOD数据集获取：
	    下载地址：https://hyper.ai/datasets/5419
	    下载解压后把数据集改名为UCAS_AOD,存储位置为UCAS_AOD_PATH
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
	5.开始训练
	    单卡训练：bash ./test/train_full_1p.sh --data_path=UCAS_AOD_PATH
	    多卡训练：bash ./test/train_full_8p.sh --data_path=UCAS_AOD_PATH
	    第一次训练如果预训练模型下载失败，可以查看下载日志，获取下载地址和存放路径，手动下载后放到对应位置
	6.获得推理结果
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