# 用于Textcnn模型离线推理指导
## 1 获取开源代码

```
git clone https://gitee.com/zhang_kaiqi/ascend-textcnn.git
cd  ascend-textcnn
git checkout 7cd94c509dc3f615a5d8f4b3816e43ad837a649e
cd -
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
cd MagicONNX && git checkout 8d62ae9dde478f35bece4b3d04eef573448411c9
cd -
```
## 2 文件放置
把TextCNN*脚本和gen_dataset_info.py脚本放到ascend_textcnn文件夹里;把*.sh脚本和fit_onnx.py放在ascend_textcnn的平行文件夹

## 3 模型推理
3.1. 前处理

```
cd ascend_textcnn
python3 TextCNN_preprocess.py --save_folder bin
python3 gen_dataset_info.py bin info
```

3.2. 转onnx

获取[TextCNN_9045_seed460473.pth](https://gitee.com/hex5b25/ascend-textcnn/raw/master/Chinese-Text-Classification-Pytorch/THUCNews/saved_dict/TextCNN_9045_seed460473.pth)

```
python3 TextCNN_pth2onnx.py --weight_path ./TextCNN_9045_seed460473.pth --onnx_path ./dy_textcnn.onnx
```

3.3. 转om

    ${chip_name}可通过`npu-smi info`指令查看
   
    ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

    该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。
    
    ```
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd ..
    bash onnxsim.sh
    bash onnx2mgonnx.sh
    bash onnx2om.sh Ascend${chip_name} # Ascend310P3
    ```

3.4. 后处理得到精度

请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

以bs1为例 

```
mkdir -p ./output_data/bs1
python3 -m ais_bench --model mg_om_dir/textcnn_1bs.om --input ./input_data --output ./output_data/bs1 --batchsize 1 --device 0
python3 TextCNN_postprocess.py ./output_data/bs1/ > result_bs1.json
```
3.5. 性能数据

推理结果打屏显示，以1bs为例

```
python3 -m ais_bench --model mg_om_dir/textcnn_1bs.om --output ./output_data/bs1 --outfmt BIN --loop 100 --device 0
```

## 4 自验
| 模型           | 官网精度   | 310P离线推理精度 | 310P性能 |
|--------------|--------|-----------|-------|
| Textcnn 64bs | [91.22%](https://gitee.com/huangyd8/Chinese-Text-Classification-Pytorch) | 90.47%    |  27242.83     |

