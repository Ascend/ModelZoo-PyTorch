from collections import defaultdict

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from ECAPA_TDNN.main import ECAPA_TDNN, load_checkpoint, inference_embeddings_to_plt_hist_and_roc
from preprocess import get_dataloader
import sys

if __name__ == "__main__":
    device = torch.device("cuda")
    model = ECAPA_TDNN(1211, device).to(device)
    checkpoint = sys.argv[1]
    data_set = sys.argv[2]
    batch_size = int(sys.argv[3])
 
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-5)

    model, optimizer, step = load_checkpoint(model, optimizer, checkpoint , rank='cpu')
    model.eval()
    acc_list = list()
    dataset_test, test_speakers = get_dataloader('vox1', 19, batch_size, data_set)
    embedding_holder = defaultdict(list)
    for mels, mel_length, speakers in tqdm(dataset_test):
        h_tensor, info_tensors = model(mels.to(device), infer=True)
        for h, s in zip(h_tensor.detach().cpu(), speakers):
            embedding_holder[s.item()].append(h.numpy())
    
    infer_hist, infer_roc, scores = inference_embeddings_to_plt_hist_and_roc(embedding_holder, step)
    tp, tn, roc_auc = scores
    print('origin roc_auc:')
    print(roc_auc)

