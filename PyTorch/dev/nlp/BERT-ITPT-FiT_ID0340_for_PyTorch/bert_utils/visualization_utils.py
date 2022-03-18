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
Script containing utilities for (attention) visualizations.
NOTE: If you are using macOS, Tkinter might crash your computer depending on your local Python,
Tkinter, and operating system versions. Check https://www.python.org/download/mac/tcltk/ for more
information.
"""

import tkinter as tk


def visualize_attention(window_name, tokens_and_weights):
    """
    Function to visualize the attention mechanism through token highlighting.

    @param (str) window_name: screen-name for the Tkinter window
    @param (list) tokens_and_weights: list containing tuples as elements with tokens and the
           corresponding attention weights, in the form (token, weight); ex. ('awesome', 0.67)
    """
    root = tk.Tk()
    root.title(window_name)
    text_widget = tk.Text(root)
    text = ''

    # List of indices, where each element will be a tuple in the form: (start_index, end_index)
    low_attention_indices = []
    medium_attention_indices = []
    high_attention_indices = []
    very_high_attention_indices = []

    # Iterate over tokens and weights and assign start and end indices depending on attention weight
    current_index = 0
    for token_and_weight in tokens_and_weights:
        token, weight = token_and_weight[0], token_and_weight[1]
        text += token + ' '

        if weight >= 0.80:
            very_high_attention_indices.append((current_index, current_index + len(token)))
        elif weight >= 0.60:
            high_attention_indices.append((current_index, current_index + len(token)))
        elif weight >= 0.40:
            medium_attention_indices.append((current_index, current_index + len(token)))
        elif weight >= 0.20:
            low_attention_indices.append((current_index, current_index + len(token)))

        current_index += len(token) + 1

    text_widget.insert(tk.INSERT, text)
    text_widget.pack(expand=1, fill=tk.BOTH)

    # Add Tkinter tags to the specified indices in text widget
    for indices in low_attention_indices:
        text_widget.tag_add('low_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    for indices in medium_attention_indices:
        text_widget.tag_add('medium_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    for indices in high_attention_indices:
        text_widget.tag_add('high_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    for indices in very_high_attention_indices:
        text_widget.tag_add('very_high_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    # Highlight attention in text based on defined tags and the corresponding indices
    text_widget.tag_config('low_attention', background='#FDA895')
    text_widget.tag_config('medium_attention', background='#FE7D61')
    text_widget.tag_config('high_attention', background='#FC5430')
    text_widget.tag_config('very_high_attention', background='#FF2D00')

    root.mainloop()
