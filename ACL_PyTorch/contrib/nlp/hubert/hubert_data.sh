#! /bin/bash
set -e

# test-clean
python3.7.5 ./fairseq/examples/wav2vec/wav2vec_manifest.py ./data/LibriSpeech/test-clean/ --dest ./data/test-clean --ext flac --valid-percent 0
python3.7.5 ./fairseq/examples/wav2vec/libri_labels.py ./data/test-clean/train.tsv --output-dir ./data/test-clean --output-name train