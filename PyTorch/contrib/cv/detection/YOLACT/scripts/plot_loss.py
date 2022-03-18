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
import re, sys, os
import matplotlib.pyplot as plt

from utils.functions import MovingAverage

with open(sys.argv[1], 'r') as f:
	inp = f.read()

patterns = {
	'train': re.compile(r'\[\s*(?P<epoch>\d+)\]\s*(?P<iteration>\d+) \|\| B: (?P<b>\S+) \| C: (?P<c>\S+) \| M: (?P<m>\S+) \|( S: (?P<s>\S+) \|)? T: (?P<t>\S+)'),
	'val': re.compile(r'\s*(?P<type>[a-z]+) \|\s*(?P<all>\S+)')
}
data = {key: [] for key in patterns}

for line in inp.split('\n'):
	for key, pattern in patterns.items():
		f = pattern.search(line)
		
		if f is not None:
			datum = f.groupdict()
			for k, v in datum.items():
				if v is not None:
					try:
						v = float(v)
					except ValueError:
						pass
					datum[k] = v
			
			if key == 'val':
				datum = (datum, data['train'][-1])
			data[key].append(datum)
			break


def smoother(y, interval=100):
	avg = MovingAverage(interval)

	for i in range(len(y)):
		avg.append(y[i])
		y[i] = avg.get_avg()
	
	return y

def plot_train(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Training Loss')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	loss_names = ['BBox Loss', 'Conf Loss', 'Mask Loss']

	x = [x['iteration'] for x in data]
	plt.plot(x, smoother([y['b'] for y in data]))
	plt.plot(x, smoother([y['c'] for y in data]))
	plt.plot(x, smoother([y['m'] for y in data]))

	if data[0]['s'] is not None:
		plt.plot(x, smoother([y['s'] for y in data]))
		loss_names.append('Segmentation Loss')

	plt.legend(loss_names)
	plt.show()

def plot_val(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Validation mAP')
	plt.xlabel('Epoch')
	plt.ylabel('mAP')

	x = [x[1]['epoch'] for x in data if x[0]['type'] == 'box']
	plt.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'box'])
	plt.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'mask'])

	plt.legend(['BBox mAP', 'Mask mAP'])
	plt.show()

if len(sys.argv) > 2 and sys.argv[2] == 'val':
	plot_val(data['val'])
else:
	plot_train(data['train'])
