# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from arguments import InitializationArguments
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser


# Configuration
parser = HfArgumentParser(InitializationArguments)
args = parser.parse_args()

# Load codeparrot tokenizer trained for Python code tokenization
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

# Config: "scale_attn_by_layer_idx" and "reorder_and_upcast_attn" are Mistral stability tweaks
config_kwargs = {"vocab_size": len(tokenizer), "scale_attn_by_layer_idx": True, "reorder_and_upcast_attn": True}

# Load model config (GPT-2 large in this case)
config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)

# Initialize new model with config
model = AutoModelForCausalLM(config)

# Save model to the hub
model.save_pretrained(args.model_name, push_to_hub=args.push_to_hub)
