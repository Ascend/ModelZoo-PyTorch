#!/bin/bash

pip install./packages/nltk-3.6.2-py3-none-any.whl

mkdir -p ~/nltk_data/corpora
mkdir -p ~/nltk_data/tokenizers

unzip -d ~/nltk_data/corpora ./packages/stopwords.zip
unzip -d ~/nltk_data/tokenizers ./packages/punkt.zip