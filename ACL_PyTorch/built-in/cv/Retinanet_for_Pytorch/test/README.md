环境准备：  

1.数据集路径  通用的数据集统一放在/root/datasets/或/opt/npu/  本模型数据集放在/root/datasets/  
该文件夹下要包含

./coco
    ./val2017
    ./annotations
         ./instances_val2017.json

2.进入工作目录  cd Retinanet 

3.安装必要的依赖 pip3.7 install -r requirements.txt

4.获取、修改和安装模型代码 a. 获取模型代码 git clone https://github.com/facebookresearch/detectron2

cd detectron2/ git reset --hard 13afb03

b.安装 cd .. rm -rf detectron2/build/ **/*.so python3.7 -m pip install -e detectron2

如果detectron2安装报错，请尝试下面这种方法安装： cd detectron2
python3.7 setup.py build develop

c.修改源代码 
若是基准环境中
cd detectron2/ patch -p1 < ../retinanet_pth.diff cd ..

若是npu环境中
cd detectron2/ patch -p1 < ../retinanet_detectron2.diff cd ..

5.获取权重文件  
wget https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl 
将权重文件改名： mv model_final_5bd44e.pkl  model_final.pkl

6.修改pytorch代码去除导出onnx时进行检查 将/usr/local/python3.7.5/lib/python3.7/site-packages/torch/onnx/utils.py文件的_check_onnx_proto(proto)这一行改为pass

7.获取benchmark工具 将benchmark.x86_64 benchmark.aarch64放在当前目录

推理步骤：

8..修改detectron2/detectron2/data/datasets/builtin.py的258行为_root = os.getenv("DETECTRON2_DATASETS", "/root/datasets/")指定coco数据集所在的目录/root/datasets/
运行命令，在output文件夹下生成model.onnx文件 

python3.7 ./detectron2/tools/deploy/export_model.py --config-file ./detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml --output ./output --export-method tracing --format onnx MODEL.WEIGHTS model_final.pkl MODEL.DEVICE cpu


9.脚本转换om模型 bash test/pth2om.sh

10.310上执行，执行时确保device空闲： bash test/eval_acc_perf.sh --datasets_path=/root/datasets
