import time
import torch
import torch_aie
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from datasets import load_dataset, load_from_disk
from transformers import ElectraForSequenceClassification, ElectraTokenizer

dataset = load_from_disk('./mrpc')
tokenizer = ElectraTokenizer.from_pretrained('electra-base-discriminator')

model_path = "electra_for_sequence_classification.pt"
model = torch.jit.load(model_path)
model.eval()

def encode(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=256)

dataset = dataset.map(encode, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

print("aie", dataset['validation'][0]['input_ids'].shape[0])

input_aie = [
    torch_aie.Input((1, dataset['validation'][0]['input_ids'].shape[0]), dtype=torch.int32),
    torch_aie.Input((1, dataset['validation'][0]['attention_mask'].shape[0]), dtype=torch.int32),
    torch_aie.Input((1, dataset['validation'][0]['token_type_ids'].shape[0]), dtype=torch.int32)
]

compiled_module = torch_aie.compile(
    model,
    inputs=input_aie,
    precision_policy=torch_aie.PrecisionPolicy.PREF_FP32,
    truncate_long_and_double=True,
    require_full_compilation=False,
    min_block_size=3,
    torch_executed_ops=[],
    soc_version="Ascend310P3",
    optimization_level=0)

torch_aie.set_device(0)

correct = 0
total = 0
start = time.time()
for batch in tqdm(dataset['validation']):
    input_ids = batch['input_ids'].unsqueeze(0).to(torch.int32)
    attention_mask = batch['attention_mask'].unsqueeze(0).to(torch.int32)
    token_type_ids = batch['token_type_ids'].unsqueeze(0).to(torch.int32)
    labels = batch['label'].unsqueeze(0).to(torch.int32)
    outputs = compiled_module.forward(input_ids, attention_mask, token_type_ids)
    pred = torch.argmax(outputs[0], dim=-1)
    correct += (pred == labels).sum().item()
    total += labels.size(0)
end = time.time()
print(f"aie inference time: {end - start}")
print("Total: ", total, "Correct: ", correct)
print(f'Accuracy: {correct / total}')

