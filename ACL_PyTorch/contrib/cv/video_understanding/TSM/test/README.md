环境准备：  

1.进入工作目录  
cd TSM

2.获取模型代码  
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .
cd ..

3.数据集准备
该模型使用 UCF-101(https://www.crcv.ucf.edu/research/data-sets/ucf101/) 的验证集进行测试，数据集下载步骤如下
cd ./mmaction2/tools/data/ucf101
bash download_annotations.sh
bash download_videos.sh
bash extract_rgb_frames_opencv.sh
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh

（可选）本项目默认将数据集存放于/opt/npu/，需要将下载下来的数据集移动至对应路径
cd ..
mv /ucf101 /opt/npu/

返回工作目录
cd TSM

4.安装必要的依赖
pip3.7 install -r requirements.txt 

5.获取权重文件  
wget https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth

6.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64放在当前目录  

7.获取msame工具
git clone https://gitee.com/ascend/tools.git
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub
cd ./tools/msame
./build.sh g++ ./
cd ../..

8.310上执行，执行时确保device空闲
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/opt/npu

9.在t4环境上将onnx_sim文件夹与perf_t4.sh放在同一目录  
然后执行bash perf_t4.sh，执行时确保gpu空闲  