# WideResnet101_2

This implements training of WideResnet101_2 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## WideResnet101_2 Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, WideResnet101_2 is re-implemented using semantics such as custom OP.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main_npu_1p.py` or `main_npu_8p.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path
```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/WideReesnet50_2_bs8192_8p_perf.log  # 8p training performance result log
    test/output/devie_id/WideReesnet50_2_bs8192_8p_acc.log   # 8p training accuracy result log

# eval default 8p， should support 1p
bash ./test/train_eval_8p.sh --data_path=real_data_path  
```

## WideResnet101_2 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 386       | 1        | 1        | O2       |
| 78.627   | 3109      | 8        | 90       | O2       |

```

## Inference
Download the mindx SDK development kit(https://www.hiascend.com/software/mindx-sdk/sdk-detail), version:2.0.2
then Compile inference image, start the docker
```
docker build -t infer_image --build-arg FROM_IMAGE_NAME=base_image:tag --build-arg SDK_PKG=sdk_pkg
bash docker_start_infer.sh docker_image model_dir
```

# mxbase
configure environment variables and modify label_file and offline_inference model path in opencv.cpp.
then, Execute the program and start inference, 
```
bash build.sh
./wideresnet [val_image_path]
```
calculate the inference accuracy
```
python3.7 classification_task_metric.py result/ ../../data/config/val_label.txt . ./result.json
cat result.json
```

# sdk
run ''' python main.py --help ''' to view the parameter details and modify them accordingly.
then, start inference and calculate the inference accuracy
```
bash run.sh ../../data/input/result
python3.7 classification_task_metric.py result/ ../../data/config/val_label.txt . ./result.json
cat result.json
```
