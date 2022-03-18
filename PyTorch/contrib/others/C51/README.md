# C51

This implements training of C51 on the game BreakoutNoFrameskip-v4, mainly modified from [pytorch/examples](https://github.com/ShangtongZhang/DeepRL).

## C51 Detail

C51 is an extension of DQN. Unlike the common methods of reinforcement learning, c51 model the expectation of that return or value.

## Requirements

- Install PyTorch and torchvision([pytorch.org](http://pytorch.org))
- `安装requirements.txt里面要求的依赖`，为了baseline安装成功，其中tensorflow版本必须大于1.14
- `conda install mpi4py`
- `git clone https://github.com/openai/baselines.git`  
  `cd baselines`  
  `pip install -e '.[all]'`


## Training

To train a model, run `train_c51.py` with the desired model architecture:

```bash
# training 1p accuracy
bash test/train_full_1p.sh 

# training 1p performance
bash test/train_performance_1p.sh

# training 1p eval
bash test/train_eval_1p.sh --pth_path=data/CategoricalDQNAgent-train_full_1p-xx.model ---status_path=data/CategoricalDQNAgent-train_full_1p-xx.stats
```

Log path:
test/output/{device_id}/train_full_1p_{device_id}.txt
test/output/{device_id}/train_performance_1p_{device_id}.txt  
test/output/{device_id}/train_eval_1p_{device_id}.txt



## C51 training result

| Acc@1    | FPS       | Npu_nums | steps   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 99.4 step/s      | 1        | 4000000        | O1       |

