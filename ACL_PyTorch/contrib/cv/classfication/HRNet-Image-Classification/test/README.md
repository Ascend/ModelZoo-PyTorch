环境准备：     

1.数据集路径  
数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集放在/opt/npu/         

2.进入工作目录  
cd HRNet         

3.安装必要的依赖  
pip3.7 install -r requirements.txt         

4.获取模型代码    
git clone https://github.com/HRNet/HRNet-Image-Classification.git  

5.获取权重文件    
wget https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/image_classification/HRNet/NPU/8P/model_best.pth.tar  
file name:model_best.pth.tar      
md5sum:1f1d61e242ac9ca4cab5d0c49299cb76  

6.获取benchmark工具    
将benchmark.x86_64 benckmark.aarch64放在当前目录  

7.310上执行               

bash test/pth2om.sh                  

bash test/eval_acc_perf.sh --datasets_path=/opt/npu        
      