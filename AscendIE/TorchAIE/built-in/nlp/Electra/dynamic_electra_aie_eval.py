import time
import torch
import torch_aie
from datasets import load_from_disk
from tqdm import tqdm
from transformers import ElectraTokenizer

def encode(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)

dataset = load_from_disk('./mrpc')
tokenizer = ElectraTokenizer.from_pretrained('electra-base-discriminator')
model_path = "electra_for_sequence_classification.pt"
model = torch.jit.load(model_path)
model.eval()

dataset = dataset.map(encode, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

compile_input = [
    torch_aie.Input(min_shape = ([1, 8]), max_shape = (1, 512), dtype=torch.int32),
    torch_aie.Input(min_shape = ([1, 8]), max_shape = (1, 512), dtype=torch.int32),
    torch_aie.Input(min_shape = ([1, 8]), max_shape = (1, 512), dtype=torch.int32)
]

compiled_module = torch_aie.compile(
    model,
    inputs=compile_input,
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
    print("id shape:", input_ids.size(), " type:", input_ids.dtype)
    print("att shape:", attention_mask.size(), " type:", attention_mask.dtype)
    print("token shape:", token_type_ids.size(), " type:", token_type_ids.dtype)
    print("labels shape:", labels.size(), " type:", labels.dtype)
    outputs = compiled_module.forward(input_ids, attention_mask, token_type_ids)
    pred = torch.argmax(outputs[0], dim=-1)
    correct += (pred == labels).sum().item()
    total += labels.size(0)

end = time.time()
print(f"aie inference time: {end - start}")
print("Total: ", total, "Correct: ", correct)
print(f'Accuracy: {correct / total}')

