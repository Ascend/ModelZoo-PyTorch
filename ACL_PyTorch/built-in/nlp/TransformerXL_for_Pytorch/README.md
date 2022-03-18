文件作用说明：
- mem_transformer.py               // 模型构建脚本
- eval.py                          // pytorch在线推理与onnx导出脚本
- om_eval.py                       // om推理验证结果脚本
- net_infer.py                     // pyACL接口脚本
- run_enwik8_base.sh               // 训练、推理入口脚本
- model.pt                         // 训练后的权重文件
- model_tsxl.onnx                  // onnx格式的模型文件
- modify_model.py                  // 修改模型的脚本
- model_sim_new.om                 // batchsize为1的离线模型
- get_out_node.py                 // 获取输出节点
- atc.sh                           // onnx模型转换om模型脚本
- transformer_xl.diff              // 文件补丁
- README.md

推理端到端步骤：

1. 下载代码仓
```shell
git clone https://github.com/kimiyoung/transformer-xl.git
git reset --hard 44781ed
```

2. 将ModleZoo获取的源码包覆盖到transformer-xl根目录中

3. 导出onnx
```shell
./run_enwik8_base.sh onnx --work_dir=workdir0-enwik8/check_point
```

4. 简化模型

   对导出的onnx模型使用onnx-simplifer工具进行简化，将模型中shape固定下来，以提升性能。

   执行命令：

```shell
python3.7 -m onnxsim model.onnx model_sim.onnx
```

6. 修改模型。

   进入om_gener目录，执行以下命令安装改图工具。
```shell
pip3.7 install .
```

   对模型进行修改，执行脚本。

```shell
python3.7 modify_model.py model_sim.onnx
```

7. 执行atc.sh脚本，将.onnx文件转为离线推理模型文件.om文件。

```shell
bash atc.sh model_sim_new.onnx model_tsxl
```

8. om离线推理命令：
```shell
./run_enwik8_base.sh om_eval --work_dir=workdir0-enwik8/check_point
```
