# FastPitch 1.1 for PyTorch

note
- please download from origin repo:
- https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch/filelists


This repository provides a script and recipe to train the FastPitch model to achieve state-of-the-art accuracy and is tested and maintained by NVIDIA.

This implements training of FastPitch on the LJ-Speech dataset, mainly modified from [pytorch/examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch).

## FastPitch Detail

[FastPitch](https://arxiv.org/abs/2006.06873) is one of two major components in a neural, text-to-speech (TTS) system:

- a mel-spectrogram generator such as [FastPitch](https://arxiv.org/abs/2006.06873) or [Tacotron 2](https://arxiv.org/abs/1712.05884), and
- a waveform synthesizer such as [WaveGlow](https://arxiv.org/abs/1811.00002) (see [NVIDIA example code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)).

The FastPitch model generates mel-spectrograms and predicts a pitch contour from raw input text.

## Requirements

- Install torch==1.5.0
- pip install -r requerements.txt
- Download the LJ-Speech dataset from https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com/train/zip/LJSpeech-1.1.zip. The complete dataset has the following structure:

```
./LJSpeech-1.1
├── mels             # (optional) Pre-calculated target mel-spectrograms; may be calculated on-line
├── metadata.csv     # Mapping of waveforms to utterances
├── pitch            # Fundamental frequency countours for input utterances; may be calculated on-line
├── README
└── wavs             # Raw waveforms
```



## Training

To train a model, run `train.py` with the desired model architecture and the path to the LJ-Speech dataset:

Before training, modify the dataset_path in these scripts.

```bash
# training 1p loss
bash ./test/train_full_1p.sh

# training 1p performance
bash ./test/train_performance_1p.sh

# training 8p loss
bash ./test/train_full_8p.sh

# training 8p performance
bash ./test/train_performance_8p.sh
```

```
Log path:
    test/output/train_full_1p.log              # 1p training result log
    test/output/train_performance_1p.log       # 1p training performance result log
    test/output/train_full_8p.log              # 8p training result log
    test/output/train_performance_8p.log       # 8p training performance result log
```



## Fastpitch training result

| Val Loss |   FPS    | Npu_nums | Epochs | AMP_Type |
| :------: | :------: | :------: | :----: | :------: |
|    -     | 2084.35  |    1     |   1    |    O1    |
|   3.69   | 18736.45 |    8     |  100   |    O1    |

