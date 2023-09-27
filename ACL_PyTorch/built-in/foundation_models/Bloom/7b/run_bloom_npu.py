import os
import time
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch_npu.contrib import transfer_to_npu


SEQ_LEN_IN = 32
SEQ_LEN_OUT = 32

DEVICE = "npu:0"
print("use " + DEVICE)
torch.npu.set_device(DEVICE)

# 使用二进制优化，消除动态shape的编译问题
torch.npu.set_compile_mode(jit_compile=False)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("./").half().to(DEVICE)

    # 优化ND NZ排布，消除transdata
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [104, 220, 221, 222, 223]:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = module.weight.data.npu_format_cast(2)
        print("soc_version:", soc_version, " is 910B, support ND")
    else:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = module.weight.data.npu_format_cast(29)
        print("soc_version:", soc_version, " is not 910B, support NZ")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = module.weight.data.npu_format_cast(2)
    return model, tokenizer


def get_random_input(tokenizer, batch, seq_len, past_key_values=None, with_attention_mask=True):
    input_ids = torch.randint(len(tokenizer), (batch, seq_len)).npu()
    input_ids[:, -1] = tokenizer.eos_token_id
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.DEVICE)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    attention_mask = torch.ones((batch, seq_len), device=input_ids.device) if with_attention_mask else None
    model_inputs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "attention_mask": attention_mask 
    }
    return model_inputs


def warm_up(model, tokenizer):
    model_inputs = get_random_input(tokenizer, 1, 4)
    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    for _ in range(5):
        model_inputs = get_random_input(tokenizer, 1, 1, outputs.past_key_values, False)
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )


#全量+增量
def full_and_incremental_test(seq_len, batch, test_cycle, model, tokenizer):
    print("start run.")
    warm_up(model, tokenizer)
    model_inputs = get_random_input(tokenizer, batch, seq_len)
    torch.npu.synchronize()
    start = time.time()
    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    torch.npu.synchronize()
    end = time.time()
    first_time = (end - start) * 1000
    print("first token:" + str(first_time) + "ms")
    sum_time = 0
    test_cycle -= 1
    for i in range(test_cycle):
        past_key_values = outputs.past_key_values
        model_inputs = get_random_input(tokenizer, batch, 1, outputs.past_key_values, False)
        torch.npu.synchronize()
        start = time.time()
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        torch.npu.synchronize()
        end = time.time()
        cur_time = (end - start) * 1000
        sum_time += cur_time
    avg_time = sum_time / test_cycle
    avg_token = sum_time / test_cycle
    response_token = first_time / sum_time
    print(f"average token:{avg_token}ms")
    print(f"response token:{response_token}ms")
    return first_time, avg_time


def main():
    model, tokenizer = load_model()

    batch_sizes = [1]
    seq_lens = [32]
    test_cases = [(bs, sq) for sq in seq_lens for bs in batch_sizes]
    for batch_size, seq_len in test_cases:

        print("---------------warm-up---------------")
        test_prompt = "Common sense questions and answers\n\nQuestion: Why do people need sleep\nFactual answer:"
        inputs_warm_up = tokenizer(test_prompt, return_tensors="pt", max_length=seq_len, 
                                    padding='max_length', truncation=True)

        with torch.no_grad():
            output = model.generate(
                inputs_warm_up.input_ids.to(DEVICE),
                max_new_tokens=seq_len,
                attention_mask=inputs_warm_up.attention_mask.to(DEVICE)
            )


        print("---------------inference---------------")

        torch.cuda.empty_cache() 
        torch.cuda.reset_peak_memory_stats(device=DEVICE)
        prompt = [
            "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
        ]
        # tokenize
        inputs = tokenizer(prompt[:batch_size], return_tensors="pt", padding='max_length', 
                            truncation=True, max_length=seq_len)
        # generate
        with torch.no_grad():
            generate_ids = model.generate(inputs.input_ids.to(DEVICE), 
                                            attention_mask=inputs.attention_mask.to(DEVICE), max_new_tokens=seq_len)
         # decode
        res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(res)


if __name__ == "__main__":
    main()