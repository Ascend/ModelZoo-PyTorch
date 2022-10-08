# VOLO

This implements training of volo_d1 on the ImageNet-2012 dataset and  token labeling, mainly modified from [sail-sg/volo](https://github.com/sail-sg/volo).

## VOLO Detail

There is an error of Col2im operator on pth2onnx, define the OP in volo.py. 
- The check of onnx should be commented out.
Example:
File "python3.8/site-packages/torch/onnx/utils.py", line 785, in _export
```bash
#if (operator_export_type is OperatorExportTypes.ONNX) and (not val_use_external_data_format):
    #try:
        #_check_onnx_proto(proto)
    #except RuntimeError as e:
        #raise CheckerError(e)
```
## Requirements

- Prepare the checkpoint of pytorch
- `pip install -r requirements.txt`
- Download the Imagenet-2012 dataset. Refer to the original repository https://github.com/rwightman/pytorch-image-models
- install MagicONNX
    ```bash
    git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
    cd MagicONNX
    pip install .
    ```
- compile msame
    reference from https://gitee.com/ascend/tools/tree/master/msame
```bash   
    git clone https://gitee.com/ascend/tools.git
    # 请根据实际情况设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # 如下为设置环境变量的示例，请将/home/HwHiAiUser/Ascend/ascend-toolkit/latest替换为Ascend 的ACLlib安装包的实际安装路径。
    export DDK_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest
    export NPU_HOST_LIB=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/acllib/lib64/stub

    cd $HOME/AscendProjects/tools/msame/
    ./build.sh g++ $HOME/AscendProjects/tools/msame/out
```

## preprocess the dataset

Because we use msame to inference, so we should preprocess origin dataset to `.bin` file.
And different batchsize should be different binary file. The command is below:

```bash
python volo_preprocess.py --src /opt/npu/val --des /opt/npu/data_bs1 --batchsize 1
python volo_preprocess.py --src /opt/npu/val --des /opt/npu/data_bs16 --batchsize 16
```
Then we get the binary dataset in `/opt/npu/data_bs1` or `/opt/npu/data_bs16` and also the label txt.The file named `volo_val_bs1.txt` or `volo_val_bs16.txt`

## Inference
```bash
# pth2om for batchsize 1
bash test/pth2om.sh d1_224_84.pth.tar volo_bs1.onnx volo_modify_bs1.onnx volo_bs1 1 "input:1,3,224,224"
# pth2om for batchsize 16
bash test/pth2om.sh d1_224_84.pth.tar volo_bs16.onnx volo_modify_bs16.onnx volo_bs16 16 "input:16,3,224,224"

# inference with batchsize 1 with performance
./msame --model "volo_bs1.om" --input "/opt/npu/data_bs1" --output "./" --outfmt TXT

# inference with batchsize 16  with performance
./msame --model "volo_bs16.om" --input "/opt/npu/data_bs16" --output "./" --outfmt TXT

# compute the val accuracy, modify the batchsize, result dir and label dir
bash eval_acc_perf.sh 1 /path/to/result /path/to/label.txt
```

## Volo inference result
| accuracy |    top1    | 
| :------: | :--------: | 
|    bs1   |   80.619   |  
|   bs16   |   82.275   |  

|batchsize| performance | average time  | average time without first  |
| :-----: | :---------: | :-----------: | :-------------------------: |
|   bs1   |  10.08fps   |    396.46ms   |            396.46ms         |
|   bs16  |  17.6fps    |   3635.25ms   |           3635.25ms         |


