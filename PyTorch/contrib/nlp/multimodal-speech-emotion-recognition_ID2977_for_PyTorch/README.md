# Multimodal Speech Emotion Recognition and Ambiguity Resolution

**�����ߣ�Publisher�������**

**Ӧ������Application Domain����Audio**

**�汾��Version����1**

**�޸�ʱ�䣨Modified�� ��2022.7.18**

**��ܣ�Framework����Pytorch 1.7**

**��������Processor�����N��910**

**Ӧ�ü���Categories����Official**

**������Description��������IEMOCAP���ݼ��Ķ�ģ̬���ʶ��ѵ������**

## Overview
Identifying emotion from speech is a non-trivial task pertaining to the ambiguous definition of emotion itself. In this work, we build light-weight multimodal machine learning models and compare it against the heavier and less interpretable deep learning counterparts. For both types of models, we use hand-crafted features from a given audio signal. Our experiments show that the light-weight models are comparable to the deep learning baselines and even outperform them in some cases, achieving state-of-the-art performance on the IEMOCAP dataset.

The hand-crafted feature vectors obtained are used to train two types of models:

1. ML-based: Logistic Regression, SVMs, Random Forest, eXtreme Gradient Boosting and Multinomial Naive-Bayes.
2. DL-based: Multi-Layer Perceptron, LSTM Classifier

This project was carried as a course project for the course CS 698 - Computational Audio taught by [Prof. Richard Mann](https://cs.uwaterloo.ca/~mannr/) at the University of Waterloo. For a more detailed explanation, please check the [report](https://arxiv.org/abs/1904.06022).

## Datasets
The [IEMOCAP](https://link.springer.com/content/pdf/10.1007%2Fs10579-008-9076-6.pdf) dataset was used for all the experiments in this work. Please refer to the [report](https://arxiv.org/abs/1904.06022) for a detailed explanation of pre-processing steps applied to the dataset.

## Requirements
All the experiments have been tested using the following libraries:
- xgboost==0.82
- torch==1.0.1.post2
- scikit-learn==0.20.3
- numpy==1.16.2
- jupyter==1.0.0
- pandas==0.24.1
- librosa==0.7.0

To avoid conflicts, it is recommended to setup a new python virtual environment to install these libraries. Once the env is setup, run `pip install -r requirements.txt` to install the dependencies.

## Instructions to run the code
1. Clone this repository by running `git clone `.
2. Go to the root directory of this project by running `cd multimodal-speech-emotion-recognition/` in your terminal.
3. Start a jupyter notebook by running `jupyter notebook` from the root of this project.
4. Run `1_extract_emotion_labels.ipynb` to extract labels from transriptions and compile other required data into a csv.
5. Run `2_build_audio_vectors.ipynb` to build vectors from the original wav files and save into a pickle file
6. Run `3_extract_audio_features.ipynb` to extract 8-dimensional audio feature vectors for the audio vectors
7. Run `4_prepare_data.ipynb` to preprocess and prepare audio + video data for experiments
8. It is recommended to train `LSTMClassifier` before running any other experiments for easy comparsion with other models later on:
  - Change `config.py` for any of the experiment settings. For instance, if you want to train a speech2emotion classifier, make necessary changes to `lstm_classifier/s2e/config.py`. Similar procedure follows for training text2emotion (`t2e`) and text+speech2emotion (`combined`) classifiers.
  - Run `python lstm_classifier.py` from `lstm_classifier/{exp_mode}` to train an LSTM classifier for the respective experiment mode (possible values of `exp_mode: s2e/t2e/combined`)
9. Run `5_audio_classification.ipynb` to train ML classifiers for audio
10. Run `5.1_sentence_classification.ipynb` to train ML classifiers for text
11. Run `5.2_combined_classification.ipynb` to train ML classifiers for audio+text

**Note:** Make sure to include correct model paths in the notebooks as not everything is relative right now and it needs some refactoring

## Results
Accuracy, F-score, Precision and Recall has been reported for the different experiments.

**Audio**

Models | Accuracy | F1 | Precision | Recall
---|---|---|---|---
RF | 55.3 | **55.8** | 56.9 | **57**
XGB | 54.8 | **55.9** | 56.8 | 56.6
SVM | 39.3 | 33.4 | 41 | 35
MNB | 30.6 | 8.8 | 12.9 | 17.2
LR | 31.6 | 12.9 | 15.1 | 19.7
MLP | 39.3 | 33.4 | 41 | 35
LSTM��GPU�� | 43.9 | 43.1 | 50.8 | 40.4
**LSTM��NPU��** | **45.7** | **42.7** | **44.8** | **44.6**
**E1** | **54.8** | 55.9 | 56.8 | **56.6**

E1: Ensemble (RF + XGB + MLP)

**Text**

Models | Accuracy | F1 | Precision | Recall
---|---|---|---|---
RF | 32.2 | 61.3 | 64.4 | 62.5
XGB | 52.3 | 51.7 | 67.7 | 47.9
SVM | 60.2 | 61.9 | 64.5 | **60.2**
MNB | 60.6 | 61 | **70.7** | 57.3
LR | 61.3 | 62.2 | 67.8 | 59.4
MLP | 62 | 62.1 | 61.8 | 64.2
LSTM��GPU�� | 51.2 | 51.3 | 56.0 | 49.8
**LSTM��NPU��** | **57.3** | **59.9** | **62.6** | **55.3**


**Audio + Text**

Models | Accuracy | F1 | Precision | Recall
---|---|---|---|---
RF | 64.9 | 65.3 | 69.3 | 65.1
XGB | 61.1 | 61.7 | 66.7 | 60.8
SVM | 62.8 | 63.1 | 63.5 | 63.9
MNB | 59.3 | 58.8 | 68.7 | 55.9
MLP | 64.1 | 66.1 | 65.1 | 67.6
LR | 62 | 62.3 | 66.3 | 60.8
LSTM��GPU�� | 64.2 | 64.7 | 66.8 | 63.9
**LSTM��NPU��** | **54.4** | **56.5** | **60.4** | **54.2**
E2 | 68.6 | **69.6** | 71.3 | **69.5**

For more details, please refer to the [report](https://arxiv.org/abs/1904.06022)

## Citation
If you find this work useful, please cite:

```
@article{sahu2019multimodal,
  title={Multimodal Speech Emotion Recognition and Ambiguity Resolution},
  author={Sahu, Gaurav},
  journal={arXiv preprint arXiv:1904.06022},
  year={2019}
}
```


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md