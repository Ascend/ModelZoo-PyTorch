import os

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


model_path = os.path.abspath('bert-large-NER')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path, torchscript=True)
model.eval()

batch_size = 1
seq_len = 512
input_ids = torch.ones([batch_size, seq_len]).to(torch.int64)
attention_mask = torch.ones([batch_size, seq_len]).to(torch.int64)
token_type_ids = torch.ones([batch_size, seq_len]).to(torch.int64)
input_data = [input_ids, attention_mask, token_type_ids]

traced_model = torch.jit.trace(model, input_data)
torch.jit.save(traced_model, 'bert_large_ner.pt')
