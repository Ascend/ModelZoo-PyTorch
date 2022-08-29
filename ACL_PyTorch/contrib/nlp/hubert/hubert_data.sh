#! /bin/bash
set -e

# test-clean
python ./fairseq/examples/wav2vec/wav2vec_manifest.py ./data/LibriSpeech/test-clean/ --dest ./data/test-clean --ext flac --valid-percent 0
python ./fairseq/examples/wav2vec/libri_labels.py ./data/test-clean/train.tsv --output-dir ./data/test-clean --output-name train


if [ ! -d "./data/pt" ]; then
  mkdir -p ./data/pt
  wget -nc -O  ./data/pt/hubert_large_ll60k_finetune_ls960.pt https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k_finetune_ls960.pt
fi