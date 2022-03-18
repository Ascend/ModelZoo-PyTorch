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

# In[ ]:


from py_tabnet.multitask import TabNetMultiTaskClassifier

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

get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')


# # Download census-income dataset

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


# # Load data and split

# In[ ]:


train = pd.read_csv(out)
target = ' <=50K'
if "Set" not in train.columns:
    train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index
test_indices = train[train.Set=="test"].index


# # Simple preprocessing
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


# # Define categorical features for categorical embeddings

# In[ ]:


unused_feat = ['Set']

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]


# # Network parameters

# In[ ]:


clf = TabNetMultiTaskClassifier(cat_idxs=cat_idxs,
                       cat_dims=cat_dims,
                       cat_emb_dim=1,
                       optimizer_fn=torch.optim.Adam,
                       optimizer_params=dict(lr=2e-2),
                       scheduler_params={"step_size":50, # how to use learning rate scheduler
                                         "gamma":0.9},
                       scheduler_fn=torch.optim.lr_scheduler.StepLR,
                       mask_type='entmax' # "sparsemax"
                      )


# # Training

# In[ ]:


NB_TASKS = 5 # Just a toy example to mimic multitask multiclassification problem

X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices].reshape(-1, 1)
y_train = np.hstack([y_train]*NB_TASKS)
# Set random labels to the last task to show how this works
y_train[:,-1] = np.random.randint(10, 15, y_train.shape[0]).astype(str)

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices].reshape(-1, 1)
y_valid = np.hstack([y_valid]*NB_TASKS)
# Set random labels to the last task to show how this works
y_valid[:,-1] = np.random.randint(10, 15, y_valid.shape[0]).astype(str)

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices].reshape(-1, 1)
y_test = np.hstack([y_test]*NB_TASKS)
# Set random labels to the last task to show how this works
y_test[:,-1] = np.random.randint(10, 15, y_test.shape[0]).astype(str)


# In[ ]:


max_epochs = 200 if not os.getenv("CI", False) else 2


# In[ ]:


clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    max_epochs=max_epochs , patience=20,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    loss_fn=[torch.nn.functional.cross_entropy]*NB_TASKS # Optional, just an example of list usage
) 


# In[ ]:


# plot losses
plt.plot(clf.history['loss'])


# In[ ]:


# plot logloss
plt.plot(clf.history['train_logloss'])
plt.plot(clf.history['valid_logloss'])


# In[ ]:


# plot learning rates
plt.plot(clf.history['lr'])


# ## Predictions

# In[ ]:


preds = clf.predict_proba(X_test)
test_aucs = [roc_auc_score(y_score=task_pred[:,1], y_true=y_test[:, task_idx])
             for task_idx, task_pred in enumerate(preds[:-1])] # don't compute on random last one

print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")
print(f"FINAL AUC SCORES FOR {dataset_name} : {test_aucs}")


# In[ ]:


predict_classes = clf.predict(X_test)


# In[ ]:


## accessing class mapping
clf.classes_


# In[ ]:


predict_classes


# In[ ]:


clf.target_mapper


# In[ ]:


ensemble_auc = roc_auc_score(y_score=np.mean(np.vstack([task_pred[:,1] for task_pred in preds]), axis=0),
                             y_true=y_test[:,0])


# In[ ]:


ensemble_auc


# # Save and load Model

# In[ ]:


# save tabnet model
saving_path_name = "./MultiTaskClassifier_1"
saved_filepath = clf.save_model(saving_path_name)


# In[ ]:


# define new model with basic parameters and load state dict weights
loaded_clf = TabNetMultiTaskClassifier()
loaded_clf.load_model(saved_filepath)


# In[ ]:


loaded_preds = loaded_clf.predict_proba(X_test)

loaded_test_auc = [roc_auc_score(y_score=task_pred[:,1], y_true=y_test[:, task_idx])
             for task_idx, task_pred in enumerate(loaded_preds[:-1])]

print(f"FINAL AUCS SCORE FOR {dataset_name} : {loaded_test_auc}")


# In[ ]:


assert(test_aucs == loaded_test_auc)


# In[ ]:


loaded_clf.predict(X_test)


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

