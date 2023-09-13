import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
from flagai.model.predictor.predictor import Predictor

import torch_npu
from torch_npu.contrib import transfer_to_npu

state_dict = "./checkpoints_in/"
model_name = 'aquila-7b'

loader = AutoLoader("lm",
                    model_dir=state_dict,
                    model_name=model_name,
                    use_cache=True)

model = loader.get_model().half().npu()
tokenizer = loader.get_tokenizer()

model.eval()
predictor = Predictor(model, tokenizer)

while True:
    user_input = input("please input ")
    if user_input.lower() == "false":
        break
    print("input is: ", user_input)
    with torch.no_grad():
        out = predictor.predict_generate_randomsample(user_input,
                                                      out_max_length=200,
                                                      top_p=0.95)
        print(f"pred is {out}")
