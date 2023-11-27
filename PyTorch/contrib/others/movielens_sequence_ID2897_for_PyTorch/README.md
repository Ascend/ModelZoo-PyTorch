# movielens_sequence


## Introduction

Using sequences of user-item interactions as an input for recommender models has a number of attractive properties.
Firstly, it recognizes that recommending the next item that a user may want to buy or see is precisely the goal we 
are trying to achieve. Secondly, it's plausible that the ordering of users' interactions carries additional 
information over and above just the identities of items they have interacted with. For example, a user is 
more likely to watch the next episode of a given TV series if they've just finished the previous episode. 
Finally, when the sequence of past interactions rather than the identity of the user is the input to a model, 
online systems can incorporate new users (and old users' new actions) in real time. They are fed to the existing 
model, and do not require a new model to be fit to incorporate new information (unlike factorization models).

Recurrent neural networks are the most natural way of modelling such sequence problems. In recommendations, 
gated recurrent units (GRUs) have been used with success in the Session-based recommendations with recurrent 
neural networks paper. Spotlight implements a similar model using LSTM units as one of its sequence representations.

## Data
Spotlight offers a slew of popular datasets, including Movielens 100K, 1M, 10M, and 20M. It also incorporates utilities 
for creating synthetic datasets. For example, generate_sequential generates a Markov-chain-derived interaction 
dataset, where the next item a user chooses is a function of their previous interactions:
```
from spotlight.datasets.synthetic import generate_sequential

# Concentration parameter governs how predictable the chain is;
# order determins the order of the Markov chain.
dataset = generate_sequential(num_users=100,
                              num_items=1000,
                              num_interactions=10000,
                              concentration_parameter=0.01,
                              order=3)
```


## Installation

### Command

```
conda install -c maciejkula -c pytorch spotlight
```

## Finetuning



### Run
```
python3 movielens_sequence.py cnn
```

### Results
```
 **GPU** 
| hash_ID                          | test_mrr | validation_mrr |
|----------------------------------|----------|----------------|
| 116446214e8e9f8777257296e6191762 | 0.0138   | 0.0093         |
| 64c3f3a23bcd7610184bdb9764003c20 | 0.0086   | 0.0075         |
| 945213785f1412f8282351c34195d526 | 0.0207   | 0.0190         |
| 35e3ad94a0ac8f5377020e05b6627451 | 0.0782   | 0.0683         |
| b345606155070e8e29653bda47633fdf | 0.0717   | 0.0656         |

 **NPU** 
| hash_ID                          | test_mrr | validation_mrr |
|----------------------------------|----------|----------------|
| 116446214e8e9f8777257296e6191762 | 0.0119   | 0.0119         |
| 64c3f3a23bcd7610184bdb9764003c20 | 0.0093   | 0.0064         |
| 945213785f1412f8282351c34195d526 | 0.0203   | 0.0187         |
| 35e3ad94a0ac8f5377020e05b6627451 | 0.0730   | 0.0661         |
| b345606155070e8e29653bda47633fdf | 0.0701   | 0.0688         |
```

# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md