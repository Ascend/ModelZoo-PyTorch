import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

class BertWrapper(torch.nn.Module):
    def __init__(self, bert_model):
        super(BertWrapper, self).__init__()
        self.model = bert_model

    def forward(self, ids, att_mask, token_ids):
        return self.model(ids, att_mask, token_ids)[0]

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
wrapped_model = BertWrapper(model)

inputs = tokenizer.encode_plus("Hello, my dog is cute", return_tensors='pt')
input_ids = inputs['input_ids'].to(torch.int32)
attention_mask = inputs['attention_mask'].to(torch.int32)
token_type_ids = inputs['token_type_ids'].to(torch.int32)

traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask, token_type_ids))
traced_model.save("bert_for_sequence_classification.pt")
