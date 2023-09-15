# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import time

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch_npu
import torch.nn as nn
from timm.models.vision_transformer import Block

from .llama import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import sample_top_p, _download

ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH,
                        "lib/libacltransformer_torch.so")
torch.classes.load_library(LIB_PATH)


class LLaMA_adapter(nn.Module):

    def __init__(self, llama_ckpt_dir, llama_tokenizer,
                 max_seq_len=512, max_batch_size=1,
                 clip_model='ViT-L/14',
                 v_embed_dim=768, v_depth=8,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 query_len=10, query_layer=31,
                 w_bias=False, 
                 w_lora=False, lora_rank=16, 
                 w_new_gate=False,
                 phase="finetune"):
        super().__init__()

        # load llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        w_bias = phase == "finetune"
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        ) # max_batch_size only affects inferenc

        self.query_len = query_len
        self.query_layer = query_layer

        # adapter query
        self.adapter_query = nn.Embedding(
            query_len * query_layer, model_args.dim)

        # tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # llama
        model_args.w_bias = w_bias
        model_args.w_lora = w_lora
        model_args.lora_rank = lora_rank
        model_args.w_new_gate = w_new_gate
        model_args.vocab_size = self.tokenizer.n_words
   
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)

        # training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # training parameters
        self.phase = phase
        self.num_layers = model_args.n_layers

        self.acl_weights = []
        self.adapter_in = []
        self.past_key_values: Optional[List[Tuple[Any, ...]]] = None

        head_size = model_args.dim // model_args.n_heads

        # acl modelV2
        self.acl_decoder_model = torch.classes.ModelTorch.ModelTorch(
            "LlamaAdapter7BDecoderModel")
        self.acl_encoder_model = torch.classes.ModelTorch.ModelTorch(
            "LlamaAdapter7BEncoderModel")

        self.acl_param_de = json.dumps({"headNum": model_args.n_heads, 
                                        "rmsNormEps": model_args.norm_eps,
                                        "dk": head_size, 
                                        "layerNum": self.num_layers})
        self.acl_param_en = json.dumps({"headNum": model_args.n_heads, 
                                        "rmsNormEps": model_args.norm_eps,
                                        "dk": head_size, 
                                        "layerNum": self.num_layers})
        
        self.acl_decoder_model.set_param(self.acl_param_de)
        self.acl_encoder_model.set_param(self.acl_param_en)

        # time stamp
        self.clip_time = 0
        self.pre_time = 0
        self.forward_time = 0
        self.post_time = 0
        self.decode_time = 0

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable param: {name}, {param.shape}, {param.dtype}")
        

    def init_acl_weight(self):
        weights = []
        weights_layer = self.state_dict()
        for i in range(self.num_layers):
            str_keys = f"llama.layers.{i}."
            weights_t = []
            atten_gate = weights_layer[str_keys + "attention.gate"]
            weights_layer[str_keys + "attention.gate"] = atten_gate.tanh().contiguous()
            weights_t.append(weights_layer[str_keys + "attention_norm.weight"].npu_format_cast(2))
            weights_t.append(weights_layer[str_keys + "attention.wq.weight"].npu_format_cast(29))
            weights_t.append(weights_layer[str_keys + "attention.wq.bias"].npu_format_cast(2))
            weights_t.append(weights_layer[str_keys + "attention.wk.weight"].npu_format_cast(29))
            weights_t.append(weights_layer[str_keys + "attention.wv.weight"].npu_format_cast(29))
            weights_t.append(weights_layer[str_keys + "attention.gate"].npu_format_cast(2))
            weights_t.append(weights_layer[str_keys + "attention.wo.weight"].npu_format_cast(29))
            weights_t.append(weights_layer[str_keys + "attention.wo.bias"].npu_format_cast(2))
            weights_t.append(weights_layer[str_keys + "ffn_norm.weight"].npu_format_cast(2))
            weights_t.append(weights_layer[str_keys + "feed_forward.w1.weight"].npu_format_cast(29))
            weights_t.append(weights_layer[str_keys + "feed_forward.w1.bias"].npu_format_cast(2))
            weights_t.append(weights_layer[str_keys + "feed_forward.w2.weight"].npu_format_cast(29))
            weights_t.append(weights_layer[str_keys + "feed_forward.w2.bias"].npu_format_cast(2))
            weights_t.append(weights_layer[str_keys + "feed_forward.w3.weight"].npu_format_cast(29))
            weights_t.append(weights_layer[str_keys + "feed_forward.w3.bias"].npu_format_cast(2))

            weights.extend(weights_t)
        self.acl_weights = weights
        self.acl_encoder_model.set_weight(self.acl_weights)
        self.acl_decoder_model.set_weight(self.acl_weights)

    def init_acl_encoder_param(self, hidden_states, adapter_in, freqs_cis, mask=None):
        hidden_states = hidden_states.half().npu().npu_format_cast(2)
        mask = mask.npu_format_cast(2)
        inputs = [hidden_states, freqs_cis, mask] + adapter_in
        return inputs

    def execute_acl_encoder(self, hidden_states, adapter_in, freqs_cis, mask=None):
        acl_input = self.init_acl_encoder_param(hidden_states, adapter_in, freqs_cis, mask)
        acl_model_out = self.acl_encoder_model.execute(acl_input, self.acl_param_en)
        self.past_key_values = tuple(zip(acl_model_out[1: self.num_layers + 1], acl_model_out[self.num_layers + 1:]))
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def init_acl_decoder_param(self, hidden_states, adapter_in, freqs_cis, mask=None):
        past_keys, past_values = map(list, zip(*self.past_key_values))
        mask = mask.npu_format_cast(2)
        inputs = [hidden_states, freqs_cis, mask] + adapter_in + past_keys + past_values
        return inputs

    def execute_acl_decoder(self, hidden_states, adapter_in, freqs_cis, mask=None):
        acl_input = self.init_acl_decoder_param(hidden_states, adapter_in, freqs_cis, mask)
        acl_model_out = self.acl_decoder_model.execute(acl_input, self.acl_param_de)
        self.past_key_values = tuple(zip(acl_model_out[1:self.num_layers + 1], acl_model_out[self.num_layers + 1:]))
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def forward(self, tokens, labels, imgs):
        visual_query = self.forward_visual(imgs)

        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, 0, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            if self.llama.vocab_size != 32000:
                print("vocab size invalid!")
                exit()
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())

        return c_loss, c_loss

    @torch.inference_mode()
    def forward_inference(self, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        freqs_cis_real = torch.view_as_real(freqs_cis).squeeze(0).npu()

        mask = None
        if seqlen == 1:
            mask = torch.zeros((1, 1, seqlen, seqlen)).type_as(h)
        else:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        
        if not self.acl_weights:
            self.init_acl_weight()

        if seqlen == 1:
            h = self.execute_acl_decoder(h, self.adapter_in, freqs_cis_real, mask)
        else:
            h = self.execute_acl_encoder(h, self.adapter_in, freqs_cis_real, mask)

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

    @torch.inference_mode()
    def generate(
        self, clip, imgs, prompts,
        max_gen_len: int = 1,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        bsz = len(prompts)
        params = self.llama.params

        s_clip = time.time()
        clip_output = clip.infer([imgs])
        visual_query = torch.from_numpy(clip_output[0]).npu()
        e_clip = time.time()

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)

        self.adapter_in = []
        for idx in range(self.query_layer):
            dynamic_adapter = adapter[idx].repeat(bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            dynamic_adapter = dynamic_adapter.half().npu()
            self.adapter_in.append(dynamic_adapter)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).npu()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).npu()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0

        empty_tuple: Tuple[Any, ...] = (None, None)
        self.past_key_values: Optional[List[Tuple[Any, ...]]] = [empty_tuple] * self.num_layers

        e_pre = time.time()
        count = 0
        for cur_pos in range(start_pos, total_len):
            s_forward = time.time()

            with torch_npu.npu.amp.autocast():
                logits = self.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos)

            e_forward = time.time()

            logits = logits.float().cpu()
            if temperature > 0:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1).npu()

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

            e_post = time.time()
            count += 1
            
            self.forward_time = (e_forward - s_forward) * 1000
            self.post_time = (e_post - e_forward) * 1000

        s_decode = time.time()
        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        e_decode = time.time()

        self.clip_time = (e_clip - s_clip) * 1000
        self.pre_time = (e_pre - e_clip) * 1000
        self.decode_time = (e_decode - s_decode) * 1000

        return decoded

_MODELS = {
    # "BIAS-7B": "", #if need, add doenload link here
    # "LORA-BIAS-7B": "",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}


def available_models():
    return list(_MODELS.keys())


def load(name, llama_dir, device="cuda" if torch.cuda.is_available() else "cpu", download_root='ckpts', max_seq_len=512,
        phase="finetune"):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}"), None

    # BIAS-7B
    llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})

    model = LLaMA_adapter(
        llama_ckpt_dir, llama_tokenzier_path,
        max_seq_len=max_seq_len, max_batch_size=1,
        clip_model='ViT-L/14',
        v_embed_dim=768, v_depth=8,
        v_num_heads=16, v_mlp_ratio=4.0,
        query_len=10, query_layer=31,
        w_bias=model_cfg.get('w_bias', False), 
        w_lora=model_cfg.get('w_lora', False), 
        lora_rank=model_cfg.get('lora_rank', 16),
        w_new_gate=model_cfg.get('w_lora', False),
        phase=phase)

    model.load_state_dict(ckpt['model'], strict=False)

    return model.half().to(device)
