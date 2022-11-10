Step1、下载如下四个文件，放置于根目录
wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model
wget https://dl.fbaipublicfiles.com/m2m_100/data_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/1.2B_last_checkpoint.pt

Step2、根目录下执行bash preprocess.sh

Step3、根目录下执行source env_npu.sh

Step4、根目录下执行bash run_1p.sh即可跑起来