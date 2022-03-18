环境准备：  

1.数据集路径  
通用的数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集放在/root/datasets/

2.进入工作目录  
cd EfficientNet-B3

3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
pip3.7 install -r requirements.txt 

4.获取模型代码  
git clone https://github.com/facebookresearch/pycls

5.如果使用补丁文件修改了模型代码则将补丁打入模型代码,如果需要引用模型代码仓的类或函数通过sys.path.append(r"./pycls")添加搜索路径。  
cd pycls   
git reset f20820e01eef7b9a47b77f13464e3e77c44d5e1f --hard   
cd ..  

6.获取权重文件  
wget https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305060/EN-B3_dds_8gpu.pyth   

7.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64放在当前目录  

8.310上执行，执行时确保device空闲  
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets
