# NasNet-A-Mobile

This implements training of Res2Net101_v1b on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/Cadene/pretrained-models.pytorch#nasnet).

# NasNet-A-Mobile Detail

# Requirements

Install PyTorch (pytorch.org)
pip install -r requirements.txt
   `Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0`
Download the ImageNet dataset from http://www.image-net.org/
Then, and move validation images to labeled subfolders, using [the following shell script]

# Training

To train a model, run main_npu_1p.py or main_npu_8p with the desired model architecture and the path to the ImageNet dataset:

# O2 training 1p
bash scripts/train_1p.sh

# O2 training 8p
bash scripts/train_8p.sh

# O2 evaling 8p
bash scripts/eval_8p.sh

# O2 online inference demo
source scripts/env_npu.sh
python3 demo.py

# O2 To ONNX
source scripts/set_npu_env.sh
python3 pthtar2onnx.py

# NasNet-A-Mobile training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 320       | 1        | 1        | O2       |
| 70.549   | 2839      | 8        | 240      | O2       |

@misc{zoph2018learning,
      title={Learning Transferable Architectures for Scalable Image Recognition}, 
      author={Barret Zoph and Vijay Vasudevan and Jonathon Shlens and Quoc V. Le},
      year={2018},
      eprint={1707.07012},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}