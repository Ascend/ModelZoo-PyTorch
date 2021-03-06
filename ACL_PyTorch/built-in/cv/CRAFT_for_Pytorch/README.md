# CARFT模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2. 安装acl_infer

   ```
   git clone https://gitee.com/peng-ao/pyacl.git
   cd pyacl
   pip3 install .
   ```

3. 获取，修改与安装开源模型代码  

   ```
   git clone https://github.com/clovaai/CRAFT-pytorch.git
   cd CRAFT-pytorch/
   git reset e332dd8b718e291f51b66ff8f9ef2c98ee4474c8 --hard
   ```

4. 下载网络权重文件craft_mlt_25k.pth

5. 数据集下载：

   使用随机数据测试余弦相似度

6. 导出onnx，生成om离线文件

   ```
   cp export_onnx.py CRAFT-pytorch/
   cp craft_mlt_25k.pth CRAFT-pytorch/
   python3 export_onnx.py --trained_model craft_mlt_25k.pth
   ```

   生成craft.onnx

7. 运行 bash craft_atc.sh生成离线om模型， craft.om

   ${chip_name}可通过`npu-smi info`指令查看
   
    ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

   ```
   cp craft_atc.sh CRAFT-pytorch/
   bash craft_atc.sh Ascend${chip_name} # Ascend310P3
   ```

   

## 2 离线推理 

1. 首先为了获得更好的性能，可以首先设置日志等级，商发版本默认ERROR级别

```
export ASCEND_GLOBAL_LOG_LEVEL=3
/usr/local/Ascend/driver/tools/msnpureport -g error -d 0
```

2. 获取精度,模型有2个输出，我们计算余弦相似度

```
cp cosine_similarity.py CRAFT-pytorch/
python3 cosine_similarity.py
```

3. [获取benchmark工具，](https://gitee.com/ascend/cann-benchmark/tree/master/infer)将benchmark.x86_64或benchmark.aarch64放到与craft相同的目录

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
./benchmark.x86_64 -round=10 -batch_size=1 -device_id=0 -om_path=craft.om 
```



```
   

|     模型      | 官网pth精度 | 310P离线推理精度 | gpu性能 | 310P性能 |
| :-----------: | :---------: | :-------------: | :-----: | :-----: |
| CRAFT_General |             |                 |         | 148fps  |

```