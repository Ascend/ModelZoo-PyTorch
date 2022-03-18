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

import re

def ptb_detokenizer(string):
	string = string.replace(" '", "'")
	string = string.replace(" \n", "\n")
	string = string.replace("\n ", "\n")
	string = string.replace(" n't", "n't")
	string = string.replace(" N ","1 ")
	string = string.replace("$ 1", "$1")
	string = string.replace("# 1", "#1")
	return string


def wikitext_detokenizer(string):
	#contractions
	string = string.replace("s '", "s'")
	string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
	# number separators
	string = string.replace(" @-@ ", "-")
	string = string.replace(" @,@ ", ",")
	string = string.replace(" @.@ ", ".")
	#punctuation
	string = string.replace(" : ", ": ")
	string = string.replace(" ; ", "; ")
	string = string.replace(" . ", ". ")
	string = string.replace(" ! ", "! ")
	string = string.replace(" ? ", "? ")
	string = string.replace(" , ", ", ")
	# double brackets
	string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
	string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
	string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
	string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
	string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
	# miscellaneous
	string = string.replace("= = = =", "====")
	string = string.replace("= = =", "===")
	string = string.replace("= =", "==")
	string = string.replace(" "+chr(176)+" ", chr(176))
	string = string.replace(" \n", "\n")
	string = string.replace("\n ", "\n")
	string = string.replace(" N ", " 1 ")
	string = string.replace(" 's", "'s")

	return string

def lambada_detokenizer(string):
	return string

def get_detokenizer(path):
	for key in DETOKENIZERS.keys():
		if key in path:
			print(key)
			return DETOKENIZERS[key]

DETOKENIZERS = {
	'ptb': ptb_detokenizer,
	'wikitext': wikitext_detokenizer,
	'lambada': lambada_detokenizer,
}
