环境准备：


1.获取开源仓代码 

    git clone https://gitee.com/hu-zongqi/modelzoo.git

2.数据集路径

通用的数据集统一放在/root/datasets/或/opt/npu/

本模型使用的数据集为Market_1501，放在目录/opt/npu/下


3.进入工作目录

    cd modelzoo/contrib/ACL_PyTorch/Research/cv/face/reid_PCB_baseline

4.下载PCB模型论文开源仓代码，并将PCB_RPP_for_reID/reid目录下的datasets文件夹移动到当前目录

    git clone https://github.com/syfafterzy/PCB_RPP_for_reID.git
    cd PCB_RPP_for_reID
    git checkout e29cf54486427d1423277d4c793e39ac0eeff87c
    cd ..
    cp -r PCB_RPP_for_reID/reid/datasets ./

5.合并补丁

    cd modelzoo/contrib/ACL_PyTorch/Research/cv/face/reid_PCB_baseline
    cp ./test/resnet.diff PCB_RPP_for_reID/reid/models
    cd PCB_RPP_for_reID/reid/models
    patch -p0 < resnet.diff

6.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装 

    cd modelzoo/contrib/ACL_PyTorch/Research/cv/face/reid_PCB_baseline
    pip3.7 install -r requirements.txt

下载onnx工具：

    cd modelzoo/contrib/ACL_PyTorch/Research/cv/face/reid_PCB_baseline/test
    git clone https://gitee.com/zheng-wengang1/onnx_tools.git scripts/utils
    cd scripts/utils
    git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921

7.获取benchmark工具

将benchmark.x86_64 放在modelzoo/contrib/ACL_PyTorch/Research/cv/face/reid_PCB_baseline目录下

8.移动PCB_RPP_for_reID/reid到modelzoo/contrib/ACL_PyTorch/Research/cv/face/reid_PCB_baseline下

    cd modelzoo/contrib/ACL_PyTorch/Research/cv/face/reid_PCB_baseline
    mv PCB_RPP_for_reID/reid ./

9.下载预训练模型文件

    cd modelzoo/contrib/ACL_PyTorch/Research/cv/face/reid_PCB_baseline/test
    mkdir models
    wget -P ./models/ https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/face/PCB/PCB_3_7.pt


10.310上执行，执行时确保device空闲 

    cd modelzoo/contrib/ACL_PyTorch/Research/cv/face/reid_PCB_baseline/test
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    sudo sh pth2om.sh 
    sudo sh eval_acc_perf.sh 
