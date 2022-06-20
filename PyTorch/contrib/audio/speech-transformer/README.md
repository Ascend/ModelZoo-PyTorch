
# Speech Transformer: End-to-End ASR with Transformer
A PyTorch implementation of Speech Transformer [1], an end-to-end automatic speech recognition with [Transformer](https://arxiv.org/abs/1706.03762) network, which directly converts acoustic features to character sequence using a single nueral network.

- Reference implementation：
```
url=https://github.com/kaituoxu/Speech-Transformer
```

## Requirements # 

- install [Kaldi](https://github.com/kaldi-asr/kaldi) (**follow the steps of option 1 in the INSTALL file**) and run some example system builds to make sure that you have kaldi installed correctly
- `cd speech-transformer/tools; make clean; make KALDI=/path/to/kaldi(where you put kaldi source code)`
- download [aishell](http://www.openslr.org/33/) dataset , decompress the data_aishell.tgz and decompress all the data in data_aishell/wav
- cd speech-transformer/test and modify data variable in the init.sh to your data_aishell path (egs: if you put data_aishell in /home and then modify data=/home)
- bash init.sh 
- Python3 (recommend Anaconda)

## Training # 
If the script has no execution permission, you need to add the permission to the script before training
```bash
cd test(be sure to run the following shell scripts in the test directory)
# 1p train perf
bash train_performance_1p.sh

# 8p train perf
bash train_performance_8p.sh

# 8p train full
bash train_full_8p.sh

# 8p eval
bash train_eval_8p.sh

# finetuning
bash train_finetune_1p.sh

# to run demo.py
cd test
source path.py # you can import kaldi_io correctly now
cd ../
python3.7 demo.py
```

## Training result # 

| CER    | Npu_nums | Epochs   | AMP_Type | FPS | 
| :------: | :------: | :------: | :------: | :------: |
| -        | 1        | 150      | O2       | 130 |
| 9.9     | 8        | 150      | O2       | 855 |
