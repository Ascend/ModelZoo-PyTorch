# RepVGG

This implements training of RepVGG on the imagenet dataset, mainly modified from [DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG).

## RepVGG Detail

For details, see (https://github.com/DingXiaoH/RepVGG)


## Requirements

- Download the ImageNet dataset refet(https://github.com/DingXiaoH/RepVGG)
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
- pip3.7 install -r requirements.txt
    - Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision
Suggestion: the pillow is 9.1.0 and the torchvision is 0.6.0
## Training

To train a model, run `train.py` with the desired model architecture and the path to the imagenet dataset:

```bash
# training 1p performance
bash test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash test/train_performance_8p.sh --data_path=real_data_path

#test 8p accuracy
bash test/train_eval_8p.sh --data_path=real_data_path

# finetune
bash test/train_finetune_1p.sh --data_path=real_data_path

# Online inference demo
python demo.py
 
```
The output log of the above script will be saved in the current folder.And the specific output log name of each script refers to the script content

## RepVGG training result

batch size 256:
| 名称      | iou      | fps      |
| :------: | :------: | :------:  | 
| GPU-1p   | -        | -      | 
| GPU-8p   | -    | 3700     | 
| NPU-1p   | -        | -      | 
| NPU-8p   | 72.08    | 600     |

batch size 2048:
| 名称      | iou      | fps      |
| :------: | :------: | :------:  | 
| GPU-1p   | -        | -      | 
| GPU-8p   | 69.60    | 6578     | 
| NPU-1p   | -        | -      | 
| NPU-8p   | 69.67    | 3265     |

batch size 4096:
| 名称      | iou      | fps      |
| :------: | :------: | :------:  | 
| GPU-1p   | -        | -      | 
| GPU-8p   | 69.41    | 8010     | 
| NPU-1p   | -        | -      | 
| NPU-8p   | 69.27    | 8596     |

