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
from matplotlib._color_data import XKCD_COLORS

with open(sys.argv[1], 'r') as f:
	txt = f.read()

txt, overall = txt.split('overall performance')

class_names = []
mAP_overall = []
mAP_small   = []
mAP_medium  = []
mAP_large   = []

for class_result in txt.split('evaluate category: ')[1:]:
	lines = class_result.split('\n')
	class_names.append(lines[0])

	def grabMAP(string):
		return float(string.split('] = ')[1]) * 100
	
	mAP_overall.append(grabMAP(lines[ 7]))
	mAP_small  .append(grabMAP(lines[10]))
	mAP_medium .append(grabMAP(lines[11]))
	mAP_large  .append(grabMAP(lines[12]))

mAP_map = {
	'small': mAP_small,
	'medium': mAP_medium,
	'large': mAP_large,
}

if len(sys.argv) > 2:
	bars = plt.bar(class_names, mAP_map[sys.argv[2]])
	plt.title(sys.argv[2] + ' mAP per class')
else:
	bars = plt.bar(class_names, mAP_overall)
	plt.title('overall mAP per class')

colors = list(XKCD_COLORS.values())

for idx, bar in enumerate(bars):
	# Mmm pseudorandom colors
	char_sum = sum([ord(char) for char in class_names[idx]])
	bar.set_color(colors[char_sum % len(colors)])

plt.xticks(rotation='vertical')
plt.show()
