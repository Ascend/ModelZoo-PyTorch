### FCENet模型PyTorch离线推理指导
###  **1. 环境准备** 

1. 安装依赖


```
pip install -r requirements.txt
```
2. 获取、安装、修改开源模型代码

```
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
patch -p1 < ../fcenet.patch
pip install -r requirements.txt
pip install -v -e . # or "python setup.py develop"
export PYTHONPATH=$(pwd):$PYTHONPATH
cd ..
```
3. 权重文件：我们利用官方的pth文件进行验证，官方pth文件可从原始开源库中获取。将权重文件fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth放到当前工作目录下。
4. 数据集：本模型需要icdar2015数据集，数据集请参考开源代码仓方式获取。获取icdar2015数据集，放到mmocr的data文件夹内,放置顺序如下。
```
├── icdar2015
│   ├── imgs
│   ├── annotations
│   ├── instances_test.json
│   └── instances_training.json
```
5. [获取msame工具](https://gitee.com/ascend/tools/tree/master/msame)
   将msame工具放到当前工作目录下。

###  **2. 离线推理**
310P上执行，执行时使npu-smi info查看设备状态，确保device空闲

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
   
```
bash test/pth2om.sh --batch_size=1 --soc_version=Ascend${chip_name} #转onnx、om模型
bash test/eval_acc_perf.sh #精度统计
```
 **评测结果** 
| 模型          | pth精度 | 310P离线推理精度 | 310P性能      |
|-------------|-------|-----------|------------|
| FCENet bs1  | 0.880 | 0.872     | fps 39.404 |


