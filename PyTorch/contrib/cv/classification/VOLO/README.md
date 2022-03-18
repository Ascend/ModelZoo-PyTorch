# VOLO

This implements training of volo_d1 on the ImageNet-2012 dataset and  token labeling, mainly modified from [sail-sg/volo](https://github.com/sail-sg/volo).

## VOLO Detail

There is an error of Col2im operator on NPU, and make it compute with the CPU. 

 Because lacking of Roi_align on NPU, the function is re-implemented .

if there is an error about `OP:ROIAlign`, please modify `/usr/local/Ascend/ascend-toolkit/5.0.x/x-linux/opp/op_impl/built-in/ai_core/tbe/impl/roi_align.py:line 2287`

```
#ori
with tik_instance.for_range(0, pool_h) as p_h:
        with tik_instance.for_range(0, pool_w, thread_num=2) as p_w:
#right
with tik_instance.for_range(0, pool_h) as p_h:
        with tik_instance.for_range(0, pool_w, thread_num=min(2, pool_w)) as p_w:
```

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))  
- `pip install -r requirements.txt`
- Download the Imagenet-2012 dataset. Refer to the original repository https://github.com/rwightman/pytorch-image-models
- Download the token label data, please refer to the [sail-sg/volo](https://github.com/sail-sg/volo).


## Training

To train a model, run `main.py` with the desired model architecture and the path to the Imagenet-2012 dataset, and :

```bash
Modify the data_dir path/to/imagenet and label path/to/label_top5_train_nfnet in the shell file.
# training 1p accuracy
bash test/train_full_1p.sh 

# training 1p performance
bash test/train_performance_1p.sh

# training 8p accuracy
bash test/train_full_8p.sh

# training 8p performance
bash test/train_performance_8p.sh

# finetune
bash test/train_finetune_1p.sh

# Online inference demo
python demo.py --checkpoint real_checkpoint_path
```

## Volo training result


|        | top1  | AMP_Type | Epochs |   FPS   |
| :----: | :---: | :------: | :----: | :-----: |
| 1p-GPU |   -   |    O2    |   1    | 152.37  |
| 1p-NPU |   -   |    O2    |   1    |  23.26  |
| 8p-GPU | 82.83 |    O2    |  100   | 1080.81 |
| 8p-NPU | 81.79 |    O2    |  100   | 180.31  |

