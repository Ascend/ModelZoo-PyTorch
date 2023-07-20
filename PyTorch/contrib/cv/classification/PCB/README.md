# PCB

This implements training of PCB on the Market-1501 dataset, mainly modified from [syfafterzy/PCB_RPP_for_reID](https://github.com/syfafterzy/PCB_RPP_for_reID).

## PCB Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, PCB is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))  

  - ~~~
    PyTorch版本：CANN 5.0.T205 PT>=20210618
    ~~~

- `pip install -r requirements.txt`
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0

- Download the Market-1501 dataset from https://paperswithcode.com/dataset/market-1501

  - ~~~shell
    unzip Market-1501-v15.09.15.zip
    ~~~

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# training 1p accuracy
bash scripts/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash scripts/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash scripts/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash scripts/train_performance_8p.sh --data_path=real_data_path

# Online inference demo
python demo.py --data_path real_data_path --device npu
## 备注： 识别前后图片保存到 `inference/` 文件夹下

# To ONNX
python pthtar2onnx.py 
	
```

## PCB training result


|        | mAP  | AMP_Type | Epochs |   FPS    |
| :----: | :--: | :------: | :----: | :------: |
| 1p-GPU |  -   |    O2    |   1    | 568.431  |
| 1p-NPU |  -   |    O2    |   1    | 571.723  |
| 8p-GPU | 77.2 |    O2    |   60   | 3600.983 |
| 8p-NPU | 77.5 |    O2    |   60   | 2750.401 |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md