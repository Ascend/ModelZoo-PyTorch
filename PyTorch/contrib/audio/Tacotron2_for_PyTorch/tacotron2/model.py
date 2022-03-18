# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sys
from os.path import abspath, dirname
# enabling modules discovery from global entrypoint
sys.path.append(abspath(dirname(__file__)+'/../'))
from common.layers import ConvNorm, LinearNorm, NpuLSTMCell
from common.utils import to_gpu, get_mask_from_lengths


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        #with torch.autograd.profiler.record_function("LocationLayer"):
        #print("attention_weights_cat:",attention_weights_cat.shape)
        processed_attention = self.location_conv(attention_weights_cat)
        #print("processed_attention:",processed_attention.shape)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim,
                 attention_dim, attention_location_n_filters,
                 attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        # self.score_mask_value = -float("inf")
        self.score_mask_value = torch.finfo(torch.float16).min

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(2)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        #with torch.autograd.profiler.record_function("Attention"):
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)
        #stream = torch.npu.current_stream()
        #stream.synchronize()
        alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        #with torch.autograd.profiler.record_function("Prenet"):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
            # x = F.relu(linear(x))
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, n_mel_channels, postnet_embedding_dim,
                 postnet_kernel_size, postnet_n_convolutions):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim, n_mel_channels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(n_mel_channels))
        )
        self.n_convs = len(self.convolutions)

    def forward(self, x):
        #with torch.autograd.profiler.record_function("Postnet"):
        i = 0
        for conv in self.convolutions:
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
                # x = torch.tanh(conv(x))
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
                # x = conv(x)
            i += 1

        return x

class DropoutV2(nn.Module):
    r"""Applies an NPU compatible dropout operation.

        This dropout method generates pseudo-random seed based on LCG(linear congruential generator) method.
        Since Ascend910 does not have a hardware unit that can generate real random numbers,
        we used the LCG method to generate pseudo-random seeds

        .. note::
            max_seed is a hyper-parameter strongly related to the underlying operator.
            Please check the MAX(2 ** 31 - 1 / 2 ** 10 - 1) in dropout_v2.py in the opp package for matching settings.
            By default, it is matched by the Pytorch and OPP packages.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = DropoutV2(p=0.5)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)
        """

    def __init__(self, p=0.5, inplace=False,
                 max_seed=2 ** 31 - 1):
        super(DropoutV2, self).__init__()

        self.p = p
        self.seed = torch.from_numpy(
            np.random.uniform(1, max_seed, size=(32 * 1024 * 12,)).astype(np.float32))

        self.checked = False

    def check_self(self, x):
        r"""Check device equipment between tensors.
        """
        if self.seed.device == x.device:
            self.checked = True
            return

        self.seed = self.seed.to(x.device)

    def forward(self, x):
        if not self.training:
            return x

        if not self.checked:
            self.check_self(x)

        x, mask, _ = torch.npu_dropoutV2(x, self.seed, p=self.p)
        return x

class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, encoder_n_convolutions,
                 encoder_embedding_dim, encoder_kernel_size):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(encoder_embedding_dim,
                         encoder_embedding_dim,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        # self.lstm = nn.LSTM(encoder_embedding_dim,
        #                     int(encoder_embedding_dim / 2), 1,
        #                     batch_first=True, bidirectional=True)
        self.lstm_fw = nn.LSTM(encoder_embedding_dim,
                               int(encoder_embedding_dim / 2), 1)
        self.lstm_bw = nn.LSTM(encoder_embedding_dim,
                               int(encoder_embedding_dim / 2), 1)

    @torch.jit.ignore
    def forward(self, x, input_lengths):
        #with torch.autograd.profiler.record_function("Encoder"):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
            # x = F.relu(conv(x))

        # x = x.transpose(1, 2)
        x = x.permute(2, 0, 1)
        #print("enter encoder*************")
        # pytorch tensor are not reversible, hence the conversion
        # input_lengths = input_lengths.cpu().numpy()
        # x = nn.utils.rnn.pack_padded_sequence(
        #     x, input_lengths, batch_first=True)
        """
        outputs = self.lstm_with_packed_sequence(x, input_lengths, bidirectional=True, batch_first=False)
        print("get output********************")
        stream = torch.npu.current_stream()
        stream.synchronize()
        outputs = outputs.transpose(0, 1)
        print("transpose*******************")
        """
        self.lstm_fw.flatten_parameters()
        outputs_fw, _ = self.lstm_fw(x)

        x_bw = torch.flip(x, [0])
        self.lstm_bw.flatten_parameters()
        outputs_bw, _ = self.lstm_bw(x_bw)
        outputs_bw = torch.flip(outputs_bw, [0])

        outputs = torch.cat((outputs_fw, outputs_bw), 2)
        outputs = outputs.transpose(0, 1)

        # outputs, _ = nn.utils.rnn.pad_packed_sequence(
        #     outputs, batch_first=True)

        return outputs

    @torch.jit.export
    def infer(self, x, input_lengths):
        device = x.device
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x.to(device))), 0.5, self.training)
            # x = F.relu(conv(x.to(device)))

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def lstmcell_with_packed_sequence(self, x, t, h0, c0, input_lengths, outputs_collect, lstm):
        #print("lstmcell_with_packed_sequence*************")
        xt = x[t, :, :].reshape(1, x.shape[1], x.shape[2])
        lstm.flatten_parameters()
        outputs, (h1, c1) = lstm(xt, (h0, c0))
        mask = (input_lengths > t).reshape(1, x.shape[1], 1)
        outputs = outputs * mask.float()
        h0 = mask.float() * h1 + (~mask).float() * h0
        c0 = mask.float() * c1 + (~mask).float() * c0
        outputs_collect.append(outputs)
        return outputs_collect, h0, c0

    def lstm_with_packed_sequence(self, x, input_lengths, bidirectional=False, batch_first=False):
        """lstm with PackedSequence in/output
           -bidirectional:If True, becomes a bidirectional LSTM
           -batch_first: If True, x: B x T x *, else x: T x B x *
        """
        print("lstm_with_packed_sequence*************")
        x = x.transpose(0, 1) if batch_first else x
        batch_dim = 1
        embed_dim = x.shape[2]
        hidden_size = int(embed_dim / 2)
        batch_size = x.shape[batch_dim]
        max_input_len = max(input_lengths)
        max_input_len = 192
        input_lengths = torch.tensor(input_lengths).to(x.device)

        if bidirectional:
            # fullfill fw LSTM
            h0_fw = torch.zeros(1, batch_size, hidden_size).to(x.device)
            c0_fw = torch.zeros(1, batch_size, hidden_size).to(x.device)
            outputs_collect = []
            for t in range(max_input_len):
                outputs_collect, h0_fw, c0_fw = self.lstmcell_with_packed_sequence(x, t, h0_fw, c0_fw, input_lengths,
                                                                                   outputs_collect, self.lstm_fw)
            stream = torch.npu.current_stream()
            stream.synchronize()
            outputs_fw = torch.cat(outputs_collect)
            #print("outputs_collect:",len(outputs_collect),outputs_collect)

            # fullfill bw LSTM
            h0_bw = torch.zeros(1, batch_size, hidden_size).to(x.device)
            c0_bw = torch.zeros(1, batch_size, hidden_size).to(x.device)
            outputs_collect = []
            for tr in range(max_input_len - 1, -1, -1):
                outputs_collect, h0_bw, c0_bw = self.lstmcell_with_packed_sequence(x, tr, h0_bw, c0_bw, input_lengths,
                                                                                   outputs_collect, self.lstm_bw)
            #import pdb
            #pdb.set_trace()
            outputs_collect.reverse()
            stream = torch.npu.current_stream()
            stream.synchronize()
            print("before cat  **************")
            #print("outputs_collect:",len(outputs_collect),outputs_collect)
            torch.save(outputs_collect,"a.tensor")
            stream = torch.npu.current_stream()
            stream.synchronize()
            outputs_bw = torch.cat(outputs_collect)

            # combine fw/bw LSTM
            stream = torch.npu.current_stream()
            stream.synchronize()
            print("before cat two *****************************")
            outputs = torch.cat((outputs_fw, outputs_bw), 2)
            print("after cat two *****************************")
        else:
            h0 = torch.zeros(1, batch_size, hidden_size).to(x.device)
            c0 = torch.zeros(1, batch_size, hidden_size).to(x.device)
            outputs_collect = []
            for t in range(max_input_len):
                outputs_collect, h0, c0 = self.lstmcell_with_packed_sequence(x, t, h0, c0, input_lengths,
                                                                             outputs_collect, self.lstm_fw)
            outputs = torch.cat(outputs_collect)
        return outputs

class Decoder(nn.Module):
    def __init__(self, n_mel_channels, n_frames_per_step,
                 encoder_embedding_dim, attention_dim,
                 attention_location_n_filters,
                 attention_location_kernel_size,
                 attention_rnn_dim, decoder_rnn_dim,
                 prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 early_stopping):
        super(Decoder, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        self.prenet = Prenet(
            n_mel_channels * n_frames_per_step,
            [prenet_dim, prenet_dim])

        #self.attention_rnn = nn.LSTMCell(
        #    prenet_dim + encoder_embedding_dim,
        #    attention_rnn_dim)
        self.attention_rnn = NpuLSTMCell(
            prenet_dim + encoder_embedding_dim,
            attention_rnn_dim)

        self.attention_layer = Attention(
            attention_rnn_dim, encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size)

        #self.decoder_rnn = nn.LSTMCell(
        #    attention_rnn_dim + encoder_embedding_dim,
        #    decoder_rnn_dim, 1)
        self.decoder_rnn = NpuLSTMCell(
            attention_rnn_dim + encoder_embedding_dim,
            decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mel_channels * n_frames_per_step)

        self.gate_layer = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

        self.attention_dropout = DropoutV2(p=p_attention_dropout)
        self.decoder_dropout = DropoutV2(p=p_decoder_dropout)

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(
            B, self.n_mel_channels*self.n_frames_per_step,
            dtype=dtype, device=device)
        return decoder_input

    def initialize_decoder_states(self, memory):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device)

        attention_cell = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device)

        decoder_hidden = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device)

        decoder_cell = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device)

        attention_weights = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device)
        attention_context = torch.zeros(
            B, self.encoder_embedding_dim, dtype=dtype, device=device)

        processed_memory = self.attention_layer.memory_layer(memory)

        return (attention_hidden, attention_cell, decoder_hidden,
                decoder_cell, attention_weights, attention_weights_cum,
                attention_context, processed_memory)

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = alignments.transpose(0, 1).contiguous()
        # (T_out, B) -> (B, T_out)
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        shape = (mel_outputs.shape[0], -1, self.n_mel_channels)
        mel_outputs = mel_outputs.view(*shape)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, attention_hidden, attention_cell,
               decoder_hidden, decoder_cell, attention_weights,
               attention_weights_cum, attention_context, memory,
               processed_memory, mask):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, attention_context), -1)
        attention_hidden, attention_cell = self.attention_rnn(
            cell_input.unsqueeze(0), (attention_hidden.unsqueeze(0), attention_cell.unsqueeze(0)))
        # attention_hidden = F.dropout(
        #     attention_hidden, self.p_attention_dropout, self.training)
        attention_hidden = self.attention_dropout(attention_hidden)

        attention_hidden, attention_cell = attention_hidden.squeeze(0), attention_cell.squeeze(0)

        attention_weights_cat = torch.cat(
            (attention_weights.unsqueeze(1),
             attention_weights_cum.unsqueeze(1)), dim=1)
        attention_context, attention_weights = self.attention_layer(
            attention_hidden, memory, processed_memory,
            attention_weights_cat, mask)

        attention_weights_cum += attention_weights
        decoder_input = torch.cat(
            (attention_hidden, attention_context), -1)

        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input.unsqueeze(0), (decoder_hidden.unsqueeze(0), decoder_cell.unsqueeze(0)))
        # decoder_hidden = F.dropout(
        #     decoder_hidden, self.p_decoder_dropout, self.training)
        decoder_hidden = self.decoder_dropout(decoder_hidden)
        decoder_hidden, decoder_cell = decoder_hidden.squeeze(0), decoder_cell.squeeze(0)

        decoder_hidden_attention_context = torch.cat(
            (decoder_hidden, attention_context), dim=1)

        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return (decoder_output, gate_prediction, attention_hidden,
                attention_cell, decoder_hidden, decoder_cell, attention_weights,
                attention_weights_cum, attention_context)

    @torch.jit.ignore
    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        #with torch.autograd.profiler.record_function("Decoder"):
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask = get_mask_from_lengths(memory_lengths)
        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell,
         attention_weights,
         attention_weights_cum,
         attention_context,
         processed_memory) = self.initialize_decoder_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []

        while len(mel_outputs) < decoder_inputs.size(0) - 1:

            decoder_input = decoder_inputs[len(mel_outputs)]
            (mel_output,
             gate_output,
             attention_hidden,
             attention_cell,
             decoder_hidden,
             decoder_cell,
             attention_weights,
             attention_weights_cum,
             attention_context) = self.decode(decoder_input,
                                              attention_hidden,
                                              attention_cell,
                                              decoder_hidden,
                                              decoder_cell,
                                              attention_weights,
                                              attention_weights_cum,
                                              attention_context,
                                              memory,
                                              processed_memory,
                                              mask)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
        #stream = torch.npu.current_stream()
        #stream.synchronize()
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            torch.stack(mel_outputs),
            torch.stack(gate_outputs),
            torch.stack(alignments))

        return mel_outputs, gate_outputs, alignments

    @torch.jit.export
    def infer(self, memory, memory_lengths):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        mask = get_mask_from_lengths(memory_lengths)
        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell,
         attention_weights,
         attention_weights_cum,
         attention_context,
         processed_memory) = self.initialize_decoder_states(memory)

        mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32, device=memory.device)
        not_finished = torch.ones([memory.size(0)], dtype=torch.int32, device=memory.device)

        mel_outputs, gate_outputs, alignments = (
            torch.zeros(1), torch.zeros(1), torch.zeros(1))
        first_iter = True
        while True:
            decoder_input = self.prenet(decoder_input)
            (mel_output,
             gate_output,
             attention_hidden,
             attention_cell,
             decoder_hidden,
             decoder_cell,
             attention_weights,
             attention_weights_cum,
             attention_context) = self.decode(decoder_input,
                                              attention_hidden,
                                              attention_cell,
                                              decoder_hidden,
                                              decoder_cell,
                                              attention_weights,
                                              attention_weights_cum,
                                              attention_context,
                                              memory,
                                              processed_memory,
                                              mask)

            if first_iter:
                mel_outputs = mel_output.unsqueeze(0)
                gate_outputs = gate_output
                alignments = attention_weights
                first_iter = False
            else:
                mel_outputs = torch.cat(
                    (mel_outputs, mel_output.unsqueeze(0)), dim=0)
                gate_outputs = torch.cat((gate_outputs, gate_output), dim=0)
                alignments = torch.cat((alignments, attention_weights), dim=0)

            dec = torch.le(torch.sigmoid(gate_output),
                           self.gate_threshold).to(torch.int32).squeeze(1)

            not_finished = not_finished*dec
            mel_lengths += not_finished

            if self.early_stopping and torch.sum(not_finished) == 0:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments, mel_lengths


class Tacotron2(nn.Module):
    def __init__(self, mask_padding, n_mel_channels,
                 n_symbols, symbols_embedding_dim, encoder_kernel_size,
                 encoder_n_convolutions, encoder_embedding_dim,
                 attention_rnn_dim, attention_dim, attention_location_n_filters,
                 attention_location_kernel_size, n_frames_per_step,
                 decoder_rnn_dim, prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 postnet_embedding_dim, postnet_kernel_size,
                 postnet_n_convolutions, decoder_no_early_stopping):
        super(Tacotron2, self).__init__()
        self.mask_padding = mask_padding
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(encoder_n_convolutions,
                               encoder_embedding_dim,
                               encoder_kernel_size)
        self.decoder = Decoder(n_mel_channels, n_frames_per_step,
                               encoder_embedding_dim, attention_dim,
                               attention_location_n_filters,
                               attention_location_kernel_size,
                               attention_rnn_dim, decoder_rnn_dim,
                               prenet_dim, max_decoder_steps,
                               gate_threshold, p_attention_dropout,
                               p_decoder_dropout,
                               not decoder_no_early_stopping)
        self.postnet = Postnet(n_mel_channels, postnet_embedding_dim,
                               postnet_kernel_size,
                               postnet_n_convolutions)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths):
        # type: (List[Tensor], Tensor) -> List[Tensor]
        if self.mask_padding and output_lengths is not None:
            mask = get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].masked_fill_(mask, 0.0)
            outputs[1].masked_fill_(mask, 0.0)
            outputs[2].masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):

        #myembedding=torch.nn.Embedding(148, 512).to("cpu")
        inputs, input_lengths, targets, max_len, output_lengths = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data
        #inputs=inputs.to("cpu").long()
        #print("inputs:",inputs.shape,inputs.dtype)
        embedded_inputs = self.embedding(inputs.long()).transpose(1, 2)
        #embedded_inputs = myembedding(inputs).transpose(1, 2)
        #embedded_inputs = embedded_inputs.half().to("npu")
        encoder_outputs = self.encoder(embedded_inputs, input_lengths)
        #print("encoder_outputs*******************")
        #import pdb
        #pdb.set_trace()
        # stream = torch.npu.current_stream()
        # stream.synchronize()
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths)
        # stream = torch.npu.current_stream()
        # stream.synchronize()
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)


    def infer(self, inputs, input_lengths):

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.infer(embedded_inputs, input_lengths)
        mel_outputs, gate_outputs, alignments, mel_lengths = self.decoder.infer(
            encoder_outputs, input_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        BS = mel_outputs_postnet.size(0)
        alignments = alignments.unfold(1, BS, BS).transpose(0,2)

        return mel_outputs_postnet, mel_lengths, alignments
