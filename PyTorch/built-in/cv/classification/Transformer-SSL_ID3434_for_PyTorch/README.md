# MoBY with Swin Transformer, Self-Supervised Pre-training and ImageNet-1K Linear Evaluation

This folder contains the implementation of the `MoBY` with `Swin Transformer` for image classification.

## Usage

### Install

- Install ASCEND-CANN, ASCEND-pytorch-1.5 and apex.

- Install `timm==0.3.2`:

```bash
pip3.7 install timm==0.3.2
```

- Install other requirements:

```bash
pip3.7 install torchvision==0.6.0 pillow==9.1.0
```

```bash
pip3.7 install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 diffdist
```

### Data preparation

We use standard ImageNet dataset.

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet 
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

### Self-Supervised Pre-training

To train `MoBY` with `Swin Transformer Tiny` on ImageNet, run:

```bash
bash ./test/train_full_8p.sh --data_path=<data_path>
```
Defaultly, training auto-resumes checkpoint in output directory. Remove the `output` directory to train from begin.

### Performance Test
For performance test on 1 NPU, run:

```bash
bash ./test/train_performance_1p.sh --data_path=<data_path>
```

For performance test on 8 NPU, run:

```bash
bash ./test/train_performance_8p.sh --data_path=<data_path>
```

### Linear Evaluation (for accuracy test)

To pretrain `MoBY` with `Swin Transformer Tiny` on ImageNet-1K for 100 epochs and do linear evaluation, run:

```bash
bash ./test/eval_8p.sh --data_path=<data_path>
```

### Training result for `MoBY Swin-T`

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type | CPU |
| :------: | :------:  | :------: | :------: | :------: |:------:|
| -        | 140       | 1        | 1        | O1       | ARM |
| 67.48    | 1150      | 8        | 100      | O1       | ARM |
| 74.14    | 1150      | 8        | 300      | O1       | ARM |