# MoBY with Swin Transformer, Self-Supervised Pre-training and ImageNet-1K Linear Evaluation

This folder contains the implementation of the `MoBY` with `Swin Transformer` for image classification.

## Usage

### Install

- Install ASCEND-CANN, ASCEND-pytorch-1.5 and apex.

- Install `timm==0.3.2`:

```bash
pip install timm==0.3.2
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 diffdist
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

For example, to train `MoBY` with `Swin Transformer Tiny` with 8 NPU on a single node for 300 epochs, run:

```bash
bash ./test/train_full_8p.sh --data_path=/data/imagenet
```

Defaultly, training auto-resumes checkpoint in output directory. Remove the `output` directory to train from begin.

### Performance Test

To train `MoBY Swin-T` on 1 NPU for performance test, run:

```bash
bash ./test/train_performance_1p.sh --data_path=<data_path>
```

For performance test on 8 NPU, run:

```bash
bash ./test/train_performance_8p.sh --data_path=<data_path>
```

### Linear Evaluation (for accuracy test)

To evaluate a pre-trained `MoBY` with `Swin Transformer Tiny` on ImageNet-1K linear evaluation, run:

```bash
bash ./test/eval_8p.sh --data_path=<data_path>
```

For example, to evaluate `MoBY Swin-T` with 8 NPU on a single node on ImageNet-1K linear evluation, run:

```bash
bash ./test/eval_8p.sh --data_path=/data/imagenet
```

### Training result for `MoBY Swin-T`

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type | CPU |
| :------: | :------:  | :------: | :------: | :------: |:------:|
| -        | 140      | 1        | 1        | O1       | ARM |
| 74.14    | 1113      | 8        | 300      | O1       | ARM |