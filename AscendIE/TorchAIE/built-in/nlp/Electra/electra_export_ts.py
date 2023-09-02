import torch
from datasets import load_dataset
from transformers import ElectraForSequenceClassification, ElectraTokenizer

class ElectraWrapper(torch.nn.Module):
    def __init__(self, electra_model):
        super(ElectraWrapper, self).__init__()
        self.model = electra_model

    def forward(self, ids, att_mask, token_ids):
        return self.model(ids, att_mask, token_ids)[0]

model = ElectraForSequenceClassification.from_pretrained('electra-base-discriminator', num_labels=2)
tokenizer = ElectraTokenizer.from_pretrained('electra-base-discriminator')
wrapper_model = ElectraWrapper(model)
wrapper_model.eval()

inputs = tokenizer("Hello, my dog is cute", return_tensors='pt')
input_ids = inputs['input_ids'].to(torch.int32)
attention_mask = inputs['attention_mask'].to(torch.int32)
token_type_ids = inputs['token_type_ids'].to(torch.int32)
traced_model = torch.jit.trace(wrapper_model, (input_ids, attention_mask, token_type_ids))

traced_model.save("electra_for_sequence_classification.pt")
