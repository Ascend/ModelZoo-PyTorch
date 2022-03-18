环境准备：  

1.数据集路径  通用的数据集统一放在/CenterFace/center-face/data

2.进入工作目录  cd CenterFace/center-face/src

3.安装模型代码之前要执行下面命令：

 git clone  https://gitee.com/Levi990223/center-face.git

4.获取权重文件  
权重文件从百度网盘上获取：https://pan.baidu.com/s/1sU3pRBTFebbsMDac-1HsQA        密码：etdi

5.获取数据集：https://www.graviti.cn/open-datasets/WIDER_FACE

6.获取benchmark工具 将benchmark.x86_64 放在CenterFace/src目录下

推理步骤：

7.调用/CenterFace/src/lib下面得pth2onnx生成onnx放在/src下面

运行python3 pth2onnx命令，在src文件夹下生成CenterFace.onnx文件 

8.脚本转换om模型 bash test/onnxToom.sh

9.310上执行，执行时确保device空闲： bash test/infer.sh 

10.在T4环境上将onnx文件放在/root/lsj目录下，执行perf_t4.sh时确保gpu空闲,执行命令 bash perf_t4.sh