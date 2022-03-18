#!/usr/bin/env python
# coding: utf-8
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#

# # Customize a TabNet Model
# 
# ## This tutorial gives examples on how to easily customize a TabNet Model
# 
# ### 1 - Customizing your learning rate scheduler
# 
# Almost all classical pytroch schedulers are now easy to integrate with pytorch-tabnet
# 
# ### 2 - Use your own loss function
# 
# It's really easy to use any pytorch loss function with TabNet, we'll walk you through that
# 
# 
# ### 3 - Customizing your evaluation metric and evaluations sets
# 
# Like XGBoost, you can easily monitor different metrics on different evaluation sets with pytorch-tabnet

# In[ ]:


from py_tabnet.tab_model import TabNetClassifier

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
np.random.seed(0)


import os
import wget
from pathlib import Path

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Download census-income dataset

# In[ ]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
dataset_name = 'census-income'
out = Path(os.getcwd()+'/data/'+dataset_name+'.csv')


# In[ ]:


out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    print("File already exists.")
else:
    print("Downloading file...")
    wget.download(url, out.as_posix())


# ### Load data and split

# In[ ]:


train = pd.read_csv(out)
target = ' <=50K'
if "Set" not in train.columns:
    train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index
test_indices = train[train.Set=="test"].index


# ### Simple preprocessing
# 
# Label encode categorical features and fill empty cells.

# In[ ]:


nunique = train.nunique()
types = train.dtypes

categorical_columns = []
categorical_dims =  {}
for col in train.columns:
    if types[col] == 'object' or nunique[col] < 200:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)


# ### Define categorical features for categorical embeddings

# In[ ]:


unused_feat = ['Set']

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]


# # 1 - Customizing your learning rate scheduler
# 
# TabNetClassifier, TabNetRegressor and TabNetMultiTaskClassifier all takes two arguments:
# - scheduler_fn : Any torch.optim.lr_scheduler should work
# - scheduler_params : A dictionnary that contains the parameters of your scheduler (without the optimizer)
# 
# ----
# NB1 : Some schedulers like torch.optim.lr_scheduler.ReduceLROnPlateau depend on the evolution of a metric, pytorch-tabnet will use the early stopping metric you asked (the last eval_metric, see 2-) to perform the schedulers updates
# 
# EX1 : 
# ```
# scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau
# scheduler_params={"mode":'max', # max because default eval metric for binary is AUC
#                  "factor":0.1,
#                  "patience":1}
# ```
# 
# -----
# NB2 : Some schedulers require updates at batch level, they can be used very easily the only thing to do is to add `is_batch_level` to True in your `scheduler_params`
# 
# EX2:
# ```
# scheduler_fn=torch.optim.lr_scheduler.CyclicLR
# scheduler_params={"is_batch_level":True,
#                   "base_lr":1e-3,
#                   "max_lr":1e-2,
#                   "step_size_up":100
#                   }
# ```
# 
# -----
# NB3: Note that you can also customize your optimizer function, any torch optimizer should work

# In[ ]:


# Network parameters
max_epochs = 20 if not os.getenv("CI", False) else 2
batch_size = 1024
clf = TabNetClassifier(cat_idxs=cat_idxs,
                       cat_dims=cat_dims,
                       cat_emb_dim=1,
                       optimizer_fn=torch.optim.Adam, # Any optimizer works here
                       optimizer_params=dict(lr=2e-2),
                       scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
                       scheduler_params={"is_batch_level":True,
                                         "max_lr":5e-2,
                                         "steps_per_epoch":int(train.shape[0] / batch_size)+1,
                                         "epochs":max_epochs
                                          },
                       mask_type='entmax', # "sparsemax",
                      )


# ### Training

# In[ ]:


X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]


# # 2 - Use your own loss function
# 
# The default loss for classification is torch.nn.functional.cross_entropy
# 
# The default loss for regression is torch.nn.functional.mse_loss
# 
# Any derivable loss function of the type lambda y_pred, y_true : loss(y_pred, y_true) should work if it uses torch computation (to allow gradients computations).
# 
# In particular, any pytorch loss function should work.
# 
# Once your loss is defined simply pass it loss_fn argument when defining your model.
# 
# /!\ : One important thing to keep in mind is that when computing the loss for TabNetClassifier and TabNetMultiTaskClassifier you'll need to apply first torch.nn.Softmax() to y_pred as the final model prediction is softmaxed automatically.
# 
# NB : Tabnet also has an internal loss (the sparsity loss) which is summed to the loss_fn, the importance of the sparsity loss can be mitigated using `lambda_sparse` parameter

# In[ ]:


def my_loss_fn(y_pred, y_true):
    """
    Dummy example similar to using default torch.nn.functional.cross_entropy
    """
    softmax_pred = torch.nn.Softmax(dim=-1)(y_pred)
    logloss = (1-y_true)*torch.log(softmax_pred[:,0])
    logloss += y_true*torch.log(softmax_pred[:,1])
    return -torch.mean(logloss)


# # 3 - Customizing your evaluation metric and evaluations sets
# 
# When calling the `fit` method you can speficy:
# - eval_set : a list of tuples like (X_valid, y_valid)
#     Note that the last value of this list will be used for early stopping
# - eval_name : a list to name each eval set
#     default will be val_0, val_1 ...
# - eval_metric : a list of default metrics or custom metrics
#     Default : "auc", "accuracy", "logloss", "balanced_accuracy", "mse", "rmse"
#     
#     
# NB : If no eval_set is given no early stopping will occure (patience is then ignored) and the weights used will be the last epoch's weights
# 
# NB2 : If `patience<=0` this will disable early stopping
# 
# NB3 : Setting `patience` to `max_epochs` ensures that training won't be early stopped, but best weights from the best epochs will be used (instead of the last weight if early stopping is disabled)

# In[ ]:


from py_tabnet.metrics import Metric


# In[ ]:


class my_metric(Metric):
    """
    2xAUC.
    """

    def __init__(self):
        self._name = "custom" # write an understandable name here
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute AUC of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            AUC of predictions vs targets.
        """
        return 2*roc_auc_score(y_true, y_score[:, 1])


# In[ ]:


clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'val'],
    eval_metric=["auc", my_metric],
    max_epochs=max_epochs , patience=0,
    batch_size=batch_size,
    virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False,
    loss_fn=my_loss_fn
) 


# In[ ]:


# plot losses
plt.plot(clf.history['loss'])


# In[ ]:


# plot auc
plt.plot(clf.history['train_auc'])
plt.plot(clf.history['val_auc'])


# In[ ]:


# plot learning rates
plt.plot(clf.history['lr'])


# ## Predictions

# In[ ]:


preds = clf.predict_proba(X_test)
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)

preds_valid = clf.predict_proba(X_valid)
valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_valid)


print(f"FINAL VALID SCORE FOR {dataset_name} : {clf.history['val_auc'][-1]}")
print(f"FINAL TEST SCORE FOR {dataset_name} : {test_auc}")


# In[ ]:


# check that last epoch's weight are used
assert np.isclose(valid_auc, clf.history['val_auc'][-1], atol=1e-6)


# # Save and load Model

# In[ ]:


# save tabnet model
saving_path_name = "./tabnet_model_test_1"
saved_filepath = clf.save_model(saving_path_name)


# In[ ]:


# define new model with basic parameters and load state dict weights
loaded_clf = TabNetClassifier()
loaded_clf.load_model(saved_filepath)


# In[ ]:


loaded_preds = loaded_clf.predict_proba(X_test)
loaded_test_auc = roc_auc_score(y_score=loaded_preds[:,1], y_true=y_test)

print(f"FINAL TEST SCORE FOR {dataset_name} : {loaded_test_auc}")


# In[ ]:


assert(test_auc == loaded_test_auc)


# # Global explainability : feat importance summing to 1

# In[ ]:


clf.feature_importances_


# # Local explainability and masks

# In[ ]:


explain_matrix, masks = clf.explain(X_test)


# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(20,20))

for i in range(3):
    axs[i].imshow(masks[i][:50])
    axs[i].set_title(f"mask {i}")


# # XGB

# In[ ]:


from xgboost import XGBClassifier

clf_xgb = XGBClassifier(max_depth=8,
    learning_rate=0.1,
    n_estimators=1000,
    verbosity=0,
    silent=None,
    objective='binary:logistic',
    booster='gbtree',
    n_jobs=-1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,)

clf_xgb.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=40,
        verbose=10)


# In[ ]:


preds = np.array(clf_xgb.predict_proba(X_valid))
valid_auc = roc_auc_score(y_score=preds[:,1], y_true=y_valid)
print(valid_auc)

preds = np.array(clf_xgb.predict_proba(X_test))
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)
print(test_auc)


# In[ ]:




