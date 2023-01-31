## 环境准备
1. 安装必要的依赖
```
pip3.7 install -r requirements.txt
```
2. 获取，修改与安装开源模型代码
```
git clone https://github.com/kimiyoung/transformer-xl.git                          # 下载原模型
git checkout 44781ed21dbaec88b280f74d9ae2877f52b492a5
git clone -b TransformerXL_large https://gitee.com/eason-hw/ModelZoo-PyTorch.git   # 下载开源仓代码
cp -r ModelZoo-PyTorch/ACL_PyTorch/contrib/nlp/TransformerXL_large/. transformer-xl/pytorch/   # 开源仓代码拷入原模型代码中
cd transformer-xl/pytorch/                               # 切换至工作目录
patch -p1 < sample.patch                                 # 如有提示"File to patch:" 需输入对应patch的文件名
```
3.获取权重文件
[model.pt](https://pan.baidu.com/s/18r4I6HC00HdMXqvPBuYJng) 提取码：so0i

4.获取数据集
```
bash getdata.sh  # enwik8数据集在data/enwik8下，包含处理好的train.txt,valid.txt和test.txt
```
5.获取安装magiconnx工具
```
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
cd MagicONNX && git checkout 8d62ae9dde478f35bece4b3d04eef573448411c9
pip install . && cd..
```

## 离线推理
310上执行时使用npu-smi info 查看设备状态，确保device空闲
1. pth转om
```
# 需要两个参数，一个是pth的路径，一个是batch_size
bash test/pth2om.sh ${pth路径} ${batch_size}
```
2. 数据集前处理
```
python3 transformerxl_large_preprocess.py --batch_size=1 --data=/data/enwik8
```
3.使用msame推理
```
# 切换至编译后的目录,--model给出om文件完整目录，--input给出输入的bin文件目录，--output给出om推理保存位置
cd tools/masme/out
./msame --model ${om文件路径} --input ${transformer-xl/pytorch/bin_data,transformer-xl/pytorch/bin_target} --output ${transformer-xl/pytorch/tools/msame/out/} --outfmt TXT 
```
4. 数据集后处理
```
# 切换回至工作目录
cd ${transformer-xl/pytorch}
bash test/eval_acc_perf.sh ${om_out_path} ${target_bin}    # om_out_path路径，target_bin路径
```
## 评测结果：
性能
| Batch Size  | A310 Throughput/Card | T4 Throughput/Card|
| ----------- | -------------------- | ----------------  |
|      1      |       6.46           |     44.1          |
|      4      |       23.67          |     49.8          |
|      8      |       41.83          |     51.9          |
|     16      |       54.98          |     49.3          |

精度(bs=16)
|   A310      |    T4     |
| ---------   | --------  |
| bpc=5.37    | bpc=5.39  |
   
