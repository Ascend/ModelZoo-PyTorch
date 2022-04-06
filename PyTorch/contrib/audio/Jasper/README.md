# Jasper

This implements training of Jasper on the LibriSpeech dataset.

- Reference implementation:
```bash
url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper
```


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the LibriSpeech dataset from http://www.openslr.org/12
   

## Training

- To run the model, you should cd to the directory of test

```bash
# 1p train full
bash ./train_full_1p.sh --data_path=xxx

#1p train perf
bash ./train_performance_1p.sh --data_path=xxx

# 8p train full
bash ./train_full_8p.sh --data_path=xxx

# 8p train perf
bash ./train_performance_8p.sh --data_path=xxx

```


## Result

Batch size 1p为32，8p为32

| 名称   | WER      | 性能/fps       | Epochs |
| :------: | :------:  | :------: | :------: |  
| GPU-1p   |    -    |   10       | 1 |     
| GPU-8p   |  10.73  |   78      | 30 |
| NPU-1p   |    -    |   4       | 1 |
| NPU-8p   |  10.89  |  34     | 30 |




