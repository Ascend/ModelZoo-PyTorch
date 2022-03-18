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
"""
Script containing various utilities related to model training, testing, and extraction of attention
weights.
"""

import logging
import torch
from bert_utils.data_utils import get_features
import time
from apex import amp


def binary_accuracy(y_pred, y_true):
    """Function to calculate binary accuracy per batch"""
    y_pred_max = torch.argmax(y_pred, dim=-1)
    correct_pred = (y_pred_max == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc


def train(model, iterator, criterion, optimizer, device, include_bert_masks=True, MAX_STEP=1):
    """
    Function to carry out the training process

    @param (torch.nn.Module) model: model object to be trained
    @param (torch.utils.data.DataLoader) iterator: data loader to iterate over batches
    @param (torch.nn.[...]) criterion: loss function to backpropagate on
    @param (torch.optim.[...]) optimizer: optimization algorithm
    @param (torch.device) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    @param (bool) include_bert_masks: whether to include token type IDs & attention masks alongside
           input IDs when passing to model or not (default: True)
    """
    epoch_loss, epoch_acc = 0.0, 0.0
    max_step = 0
    #https://gitee.com/ascend/pytorch/issues/I3TWK4?from=project-issue
    model.train()
    #https://gitee.com/ascend/pytorch/issues/I3TWK4?from=project-issue

    for batch in iterator:
        start = time.time()
        # Get training input IDs & labels from the current batch
        input_ids, labels = batch
        # Get corresponding additional features from the current batch
        token_type_ids, attention_mask = get_features(input_ids=input_ids,
                                                      tokenizer=model.get_tokenizer(),
                                                      device=device)
        # Reset the gradients from previous processes
        optimizer.zero_grad()
        # Pass features through the model w/ or w/o BERT masks for attention & token type
        if include_bert_masks:
            predictions = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        else:
            predictions = model(input_ids=input_ids)

        # Calculate loss and accuracy
        #predictions,labels=predictions.npu(),labels.npu()
        #labels = labels.to(torch.int)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        # Backward and optimize
        # 使能混合精度
        #loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
             scaled_loss.backward()
        # 使能混合精度

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        print(f'\tTrain-Loss: {epoch_loss / len(iterator):.3f} | '
              f'Train-Accuracy: {epoch_acc / len(iterator) * 100:.2f}% | '
              f'Train-Time: {time.time() - start:.6f}')
        max_step += 1
        if max_step >= MAX_STEP:
            return epoch_loss / len(iterator), epoch_acc / len(iterator)


    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def test(model, iterator, criterion, device, include_bert_masks=True, MAX_STEP=1):
    """
    Function to carry out the testing (or validation) process

    @param (torch.nn.Module) model: model object to be trained
    @param (torch.utils.data.DataLoader) iterator: data loader to iterate over batches
    @param (torch.nn.[...]) criterion: loss function to backpropagate on
    @param (torch.device) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    @param (bool) include_bert_masks: whether to include token type IDs & attention masks alongside
           input IDs when passing to model or not (default: True)
    """
    epoch_loss, epoch_acc = 0.0, 0.0
    max_step = 0

    with torch.no_grad():
        for batch in iterator:
            start = time.time()
            # Get testing input IDs & labels from the current batch
            input_ids, labels = batch
            # Get corresponding additional features from the current batch
            token_type_ids, attention_mask = get_features(input_ids=input_ids,
                                                          tokenizer=model.get_tokenizer(),
                                                          device=device)
            # Pass features through the model w/ or w/o BERT masks for attention & token type
            if include_bert_masks:
                predictions = model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
            else:
                predictions = model(input_ids=input_ids)

            # Calculate loss and accuracy
            #labels = labels.to(torch.int)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            print(f'\tTest-Loss: {epoch_loss / len(iterator):.3f} | '
                  f'Test-Accuracy: {epoch_acc / len(iterator) * 100:.2f}% | '
                  f'Test-Time: {time.time() - start:.6f}')
            max_step += 1
            if max_step >= MAX_STEP:
                return epoch_loss / len(iterator), epoch_acc / len(iterator)


    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def get_attention_nth_layer_mth_head_kth_token(attention_outputs, n, m, k, average_heads=False):
    """
    Function to compute attention weights by:
    i)   Take the attention weights from the nth multi-head attention layer assigned to kth token
    ii)  Take the mth attention head
    """
    if average_heads is True and m is not None:
        logging.warning("Argument passed for param @m will be ignored because of head averaging.")

    # Get the attention weights outputted by the nth layer
    attention_outputs_concatenated = torch.cat(attention_outputs, dim=0)       # (K, N, P, P)
    attention_outputs = attention_outputs_concatenated.data[n, :, :, :]        # (N, P, P)

    # Get the attention weights assigned to kth token
    attention_outputs = attention_outputs[:, k, :]                             # (N, P)

    # Compute the average attention weights across all attention heads
    if average_heads:
        attention_outputs = torch.sum(attention_outputs, dim=0)                # (P)
        num_attention_heads = attention_outputs_concatenated.shape[1]
        attention_outputs /= num_attention_heads
    # Get the attention weights of mth head
    else:
        attention_outputs = attention_outputs[m, :]                            # (P)

    return attention_outputs


def get_attention_average_first_layer(attention_outputs):
    """
    Function to compute attention weights by:
    i)   Take the attention weights from the first multi-head attention layer assigned to CLS
    ii)  Average each token across attention heads
    """
    return get_attention_nth_layer_mth_head_kth_token(attention_outputs=attention_outputs,
                                                      n=0, m=None, k=0,
                                                      average_heads=True)


def get_attention_average_last_layer(attention_outputs):
    """
    Function to compute attention weights by
    i)   Take the attention weights from the last multi-head attention layer assigned to CLS
    ii)  Average each token across attention heads
    """
    return get_attention_nth_layer_mth_head_kth_token(attention_outputs=attention_outputs,
                                                      n=-1, m=None, k=0,
                                                      average_heads=True)


def get_normalized_attention(model, raw_sentence, method='last_layer_heads_average',
                             n=None, m=None, k=None, exclude_special_tokens=True,
                             normalization_method='normal', device='cpu'):
    """
    Function to get the normalized version of the attention output of a FineTunedBert() model

    @param (torch.nn.Module) model: FineTunedBert() model to visualize attention weights on
    @param (str) raw_sentence: sentence in string format, preferably from the test distribution
    @param (str) method: method name specifying the attention output configuration, possible values
           are 'first_layer_heads_average', 'last_layer_heads_average', 'nth_layer_heads_average',
           'nth_layer_mth_head', and 'custom' (default: 'last_layer_heads_average')
    @param (int) n: layer no. (default: None)
    @param (int) m: head no. (default: None)
    @param (int) k: token no. (default: None)
    @param (bool) exclude_special_tokens: whether to exclude special tokens such as [CLS] and [SEP]
           from attention weights computation or not (default: True)
    @param (str) normalization_method: the normalization method to be applied on attention weights,
           possible values include 'min-max' and 'normal' (default: 'normal')
    @param (torch.device) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    """
    if None in [n, m, k] and method == 'custom':
        raise ValueError("Must pass integer argument for params @n, @m, and @k " +
                         "if method is 'nth_layer_mth_head_kth_token'")
    elif None not in [n, m, k] and method != 'custom':
        logging.warning("Arguments passed for params @n, @m, or @k will be ignored. " +
                        "Specify @method as 'nth_layer_mth_head_kth_token' to make them effective.")

    # Plug in CLS & SEP special tokens for identification of start & end points of sequences
    if '[CLS]' not in raw_sentence and '[SEP]' not in raw_sentence:
        tokenized_text = ['[CLS]'] + model.get_tokenizer().tokenize(raw_sentence) + ['[SEP]']
    else:
        tokenized_text = model.get_tokenizer().tokenize(raw_sentence)

    # Call model evaluation as we don't want no gradient update
    model.eval()
    with torch.no_grad():
        attention_outputs = model.get_bert_attention(raw_sentence=raw_sentence, device=device)

    attention_weights = None
    if method == 'first_layer_heads_average':
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=0, m=None, k=0,
            average_heads=True
        )
    elif method == 'last_layer_heads_average':
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=-1, m=None, k=0,
            average_heads=True
        )
    elif method == 'nth_layer_heads_average':
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n, m=None, k=0,
            average_heads=True
        )
    elif method == 'nth_layer_mth_head':
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n, m=m, k=0,
            average_heads=False
        )
    elif method == 'custom':
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n, m=m, k=k,
            average_heads=False
        )

    # Remove the beginning [CLS] & ending [SEP] tokens for better intuition
    if exclude_special_tokens:
        tokenized_text, attention_weights = tokenized_text[1:-1], attention_weights[1:-1]

    # Apply normalization methods to attention weights
    # i)  Min-Max Normalization
    if normalization_method == 'min-max':
        max_weight, min_weight = attention_weights.max(), attention_weights.min()
        attention_weights = (attention_weights - min_weight) / (max_weight - min_weight)

    # ii) Z-Score Normalization
    elif normalization_method == 'normal':
        mu, std = attention_weights.mean(), attention_weights.std()
        attention_weights = (attention_weights - mu) / std

    # Convert tensor to NumPy array
    attention_weights = attention_weights.data

    tokens_and_weights = []
    for index, token in enumerate(tokenized_text):
        tokens_and_weights.append((token, attention_weights[index].item()))

    return tokens_and_weights


def get_delta_attention(tokens_and_weights_pre, tokens_and_weights_post):
    """Function to compute the delta (change) in scaled attention weights before & after"""
    tokens_and_weights_delta = []
    for i, token_and_weight in enumerate(tokens_and_weights_pre):
        token,  = token_and_weight[0],
        assert token == tokens_and_weights_post[i][0]

        pre_weight = token_and_weight[1]
        post_weight = tokens_and_weights_post[i][1]

        tokens_and_weights_delta.append((token, post_weight - pre_weight))

    return tokens_and_weights_delta
