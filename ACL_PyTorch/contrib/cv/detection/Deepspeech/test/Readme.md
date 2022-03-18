1. 获取源代码
```
git clone https://github.com/SeanNaren/deepspeech.pytorch.git -b V3.0
```
2. 修改deepspeech.pytorch/deepspeech_pytorch/model.py
3. 下载ckpt权重文件
```
wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4_pretrained_v3.ckpt
```
4. 获取数据集
```
cd deepspeech.pytorch/data
python3 an4.py
cd ../..
```
5. 生成om文件
```
bash test/ckpt2om.sh
```
6. 获取msame工具，将msame放在根目录下
7. 评估精度
```
bash test/eval_acc.sh
```
8.T4性能（测试pytorch模型的在线推理性能）
在T4上获取源代码，配置相关环境，运行t4_perf.py，获取在线推理性能。

备注：
benchmark暂不支持多输入，因此改用msame工具，该工具不支持多batch，因此以bs1为准。