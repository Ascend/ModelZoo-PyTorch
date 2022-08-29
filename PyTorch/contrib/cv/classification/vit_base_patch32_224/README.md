## Vit_base_patch32_224<br>
This implements training of Vit_base_patch32_224 on the ImageNet dataset<br>
## Vit_base_patch32_224 Detail<br>
This model is one of open-source models by rwightman. See the source code at https://github.com/rwightman/pytorch-image-models/tree/master/timm.<br>
The whole model has achieved the requirement of accuracy and performance.<br>
## requirement<br>
Install PyTorch<br>
```pip install timm==0.4.12```<br>
```torchvision==0.5.0(x86) && torchvision==0.2.0(arm)```
Please prepare the dataset by yourself, including training set and verification set. The optional dataset includes imagenet2012, including train and val.
## Training<br>
To train a model, run ```vit_train.py``` with the desired model architecture and the path to the ImageNet dataset:<br>
```bash
# training 1p performance
bash test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash test/train_performance_8p.sh --data_path=real_data_path

# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=real_data_path --model-path=real_pre_train_model_path
```
Log path: output/devie_id/train_${device_id}.log<br>
## Vit_base_patch32_224 Detail<br>
| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 96.8      | 1        | 1        | O1       |
| 80.64    | 2981      | 8        | 8        | O1       |