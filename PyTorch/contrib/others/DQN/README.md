# DQN

This implements training of DQN on the game BreakoutNoFrameskip-v4, mainly modified from [pytorch/examples](https://github.com/ShangtongZhang/DeepRL).

## DQN Detail

Deep Q-Learning (DQN) combines the method of neural network and Q learning.

## Requirements

- Install PyTorch and torchvision([pytorch.org](http://pytorch.org))

- `安装requirements.txt里面要求的依赖`

- `配置mujoco`

  `从https://www.roboti.us/license.html上点击Activation key下载mjkey.txt`

  `从https://www.roboti.us/download/mjpro150_linux.zip下载mjpro150_linux.zip`

  `在root目录下创建隐藏文件夹.mujoco(不要忘记带.)，并将mjpro150_linux.zip安装包解压到这个文件夹下`

  `将mjkey.txt移动到~/.mujoco 和 ~/.mujoco/mjpro150/bin下`

  `添加环境变量, 打开～/.bashrc文件,添加以下指令`

  `export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

  `export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}`

- `baselines安装`

  `git clone https://github.com/openai/baselines.git`  

  `cd baselines`  

  `pip install -e .`


## Training

To train a model, run `train_dqn.py` with the desired model architecture:

```bash
# training 1p accuracy
bash test/train_full_1p.sh 

# training 1p performance
bash test/train_performance_1p.sh

# training 1p eval
bash test/train_eval_1p.sh --pth_path=data/DQNAgent-train_full_1p-xx.model ---status_path=data/DQNAgent-train_full_1p-xx.stats
```

Log path:

test/output/{device_id}/train_full_1p_{device_id}.txt

test/output/{device_id}/train_performance_1p_{device_id}.txt 
 
test/output/{device_id}/train_eval_1p_{device_id}.txt



## DQN training result

| Acc@1    | FPS       | Npu_nums | steps   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        |62.9 step/s| 1        | 80000    | O1       |
| 99.3     | -         | 1        | 5000000  | O1       |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md