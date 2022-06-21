# Self-Supervised Vision Transformers with DINO

This folder contains the implementation of the `DINO` for image classification.

## Usage

### Install

- Install ASCEND-CANN, ASCEND-pytorch-1.5 and apex.

- Install `timm==0.4.5`:

```bash
pip3.7 install timm==0.4.5
```

- Install other requirements:

```bash
pip3.7 install torchvision==0.6.0 pillow==9.1.0
```
Note: torchvision package should be built from source with branch v0.6.0 in ARM structure.

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

To train `DINO` on ImageNet, run:

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

### Full Test (for accuracy test)

For full test on 8 NPU, run:

```bash
bash ./test/train_full_8p.sh --data_path=<data_path>
```

### Training result for `DINO`

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type | CPU |
| :------: | :------:  | :------: | :------: | :------: |:------:|
| -        | 183      | 1        | 1        | O1       | ARM |
| -        | 1393      | 8        | 1        | O1       | ARM |
| -        | 190       | 1        | 1        | O1       | X86 |
| -        | 1329      | 8        | 1        | O1       | X86 |
| 69.7     | 1393      | 8        | 100      | O1       | ARM |

Note: ARM with 192 CPUs, X86 is Intel(R) Xeon(R) Platinum 8260 with 96 CPUs.