# DIN/DIEN 

Implementation based on pytorch for DIN recommendation algorithm


## Attention

1. For convenience, referring to authors tensorflow implementation, feature-embedding dimension is identical.
2. Without any L1/L2 normalization or dropout strategy, it's supposed to choose suitable model according to the evaluation stage manually.

## File description 
|file name|description|
|--|----|
|main.ipynb|Session for training and evaluation|
|model.py|Defination of target models|
|DataLoader.py|Self-defined data loader|
|environment.yml|Conda envrionment yaml|

## Original paper
[Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)

[Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)

## Source data
[meta_Books.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz)

[reviews_Books.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz)

Preprocessed data wrapped within `data.tar.gz` came from [mouna99/dien](https://github.com/mouna99/dien)

## Reference

[mouna99/dien](https://github.com/mouna99/dien)

[alibaba/x-deeplearning](https://github.com/alibaba/x-deeplearning)

[shenweichen/DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)


## To do list

- [x] DIN
- [x] AUGRU 
- [ ] DICE activation layer 
- [ ] Auxialary loss with neg_sample  