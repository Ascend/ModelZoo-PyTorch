# 用于Textcnn模型离线推理指导
## 1 获取开源代码

```
https://gitee.com/zhang_kaiqi/ascend-textcnn.git
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
1. 前处理

```
cd ascend_textcnn
python3 TextCNN_preprocess.py --save_folder bin
python3 gen_dataset_info.py bin info
```

2. 转onnx

获取[TextCNN_9045_seed460473.pth](https://gitee.com/hex5b25/ascend-textcnn/raw/master/Chinese-Text-Classification-Pytorch/THUCNews/saved_dict/TextCNN_9045_seed460473.pth)

```
python3 TextCNN_pth2onnx.py --weight_path ./TextCNN_9045_seed460473.pth --onnx_path ./dy_textcnn.onnx
```

3. 转om

```
cd ..
bash onnxsim.sh
bash onnx2mgonnx.sh
bash onnx2om.sh
```

4. 后处理得到精度

精度结果保存在result_bs*.json中。*代表具体的batch_size值（从4开始）

```
./benchmark.x86_64 -batch_size=* -om_path=mg_om_dir/textcnn_*bs_mg.om -output_binary=True -input_text_path=ascend-textcnn/info -useDvpp=False -model_type=nlp
python3 ascend-textcnn/TextCNN_postprocess.py result/dumpOutput_device0 >result_bs*.json
```
5. 性能数据

推理结果打屏显示

```
./msame --model mg_om_dir/trextcnn_*bs_mg.om --loop 100
```

## 3 自验
| 模型           | 官网精度   | 710离线推理精度 | 710性能 |
|--------------|--------|-----------|-------|
| Textcnn 64bs | [91.22%](https://gitee.com/huangyd8/Chinese-Text-Classification-Pytorch) | 90.47%    |  27242.83     |

