import numpy as np

import torch
from torch import optim

import random
from copy import deepcopy

from utils import get_data, ndcg, recall
from model import VAE
import time
import apex
from apex import amp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--hidden-dim', type=int, default=600)
parser.add_argument('--latent-dim', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=500)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n-epochs', type=int, default=50)
parser.add_argument('--n-enc_epochs', type=int, default=3)
parser.add_argument('--n-dec_epochs', type=int, default=1)
parser.add_argument('--not-alternating', type=bool, default=False)
# Mixed precision training parameters
parser.add_argument('--apex', action='store_true',
                    help='Use apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O1', type=str,
                    help='For apex mixed precision training'
                         'O0 for FP32 training, O1 for mixed precision training.'
                         'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')
parser.add_argument('--loss-scale-value', default=1024., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
## for ascend 910
parser.add_argument('--device_id', default=5, type=int, help='device id')
args = parser.parse_args()

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#device = torch.device("npu")
device = torch.device(f'npu:{args.device_id}')
torch.npu.set_device(device)
print("Use NPU: {} for training".format(args.device_id))

data = get_data(args.dataset)
train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = data


def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1
    
    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)
    
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
    
    def get_idx(self):
        return self._idx
    
    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)
        
    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
    
    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device,non_blocking=True)


def evaluate(model, data_in, data_out, metrics, samples_perc_per_epoch=1, batch_size=500):
    metrics = deepcopy(metrics)
    model.eval()
    
    for m in metrics:
        m['score'] = []
    
    for batch in generate(batch_size=batch_size,
                          device=device,
                          data_in=data_in,
                          data_out=data_out,
                          samples_perc_per_epoch=samples_perc_per_epoch
                         ):
        
        ratings_in = batch.get_ratings_to_dev()
        ratings_out = batch.get_ratings(is_out=True)
    
        ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()\
        #ratings_pred = model.forward(ratings_in, calculate_loss=False).cpu().detach().numpy()
        
        if not (data_in is data_out):
            ratings_pred[batch.get_ratings().nonzero()] = -np.inf
            
        for m in metrics:
            m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))

    for m in metrics:
        m['score'] = np.concatenate(m['score']).mean()
        
    return [x['score'] for x in metrics]


def run(model, opts, train_data, batch_size, n_epochs, beta, gamma, dropout_rate):
    model.train()
    step=0
    import time
    #if args.apex:
    #    model, opts = amp.initialize(model, opts,
    #                                      opt_level=args.apex_opt_level,
    #                                      loss_scale=args.loss_scale_value,
    #                                      combine_grad=True)
    for epoch in range(n_epochs):
        for batch in generate(batch_size=batch_size, device=device, data_in=train_data, shuffle=True):
            ratings = batch.get_ratings_to_dev()
            start_time = time.time()
            for optimizer in opts:
                optimizer.zero_grad()
            step+=1    
            _, loss = model(ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            for optimizer in opts:
                optimizer.step()
            #loss.backward()
            print('epoch: {}, step:{} loss: {:.2f}, time: {:.3f}'.format(epoch, step, loss, time.time() - start_time))            
            for optimizer in opts:
                optimizer.step()


model_kwargs = {
    'hidden_dim': args.hidden_dim,
    'latent_dim': args.latent_dim,
    'input_dim': train_data.shape[1]
}
metrics = [{'metric': ndcg, 'k': 100}]

best_ndcg = -np.inf
train_scores, valid_scores = [], []

model = VAE(**model_kwargs).to(device)
model = model.to(device)
model_best = VAE(**model_kwargs).to(device)

learning_kwargs = {
    'model': model,
    'train_data': train_data,
    'batch_size': args.batch_size,
    'beta': args.beta,
    'gamma': args.gamma
}

decoder_params = set(model.decoder.parameters())
encoder_params = set(model.encoder.parameters())

optimizer_encoder =apex.optimizers.NpuFusedAdam(encoder_params, lr=args.lr)
optimizer_decoder =apex.optimizers.NpuFusedAdam(decoder_params, lr=args.lr)

if args.apex:
    [model.decoder, model.encoder], [optimizer_decoder, optimizer_encoder] = amp.initialize([model.decoder, model.encoder], [optimizer_decoder, optimizer_encoder],
                                opt_level=args.apex_opt_level,
                                loss_scale=args.loss_scale_value,
                                combine_grad=True)

for epoch in range(args.n_epochs):
    #starttime=time.time()
    #steptime=time.time()

    if args.not_alternating:
        run(opts=[optimizer_encoder, optimizer_decoder], n_epochs=1, dropout_rate=0.5, **learning_kwargs)
    else:
        run(opts=[optimizer_encoder], n_epochs=args.n_enc_epochs, dropout_rate=0.5, **learning_kwargs)
        model.update_prior()
        run(opts=[optimizer_decoder], n_epochs=args.n_dec_epochs, dropout_rate=0, **learning_kwargs)
    #print("FPS:",time.time()-steptime)

    train_scores.append(
        evaluate(model, train_data, train_data, metrics, 0.01)[0]
    )
    valid_scores.append(
        evaluate(model, valid_in_data, valid_out_data, metrics, 1)[0]
    )
    
    if valid_scores[-1] > best_ndcg:
        best_ndcg = valid_scores[-1]
        model_best.load_state_dict(deepcopy(model.state_dict()))
        

    print(f'epoch {epoch} | valid ndcg@100: {valid_scores[-1]:.4f} | ' +
          f'best valid: {best_ndcg:.4f} | train ndcg@100: {train_scores[-1]:.4f}')
    #print("Training:",time.time()-starttime)



    
test_metrics = [{'metric': ndcg, 'k': 100}, {'metric': recall, 'k': 20}, {'metric': recall, 'k': 50}]

final_scores = evaluate(model_best, test_in_data, test_out_data, test_metrics)

for metric, score in zip(test_metrics, final_scores):
    print(f"{metric['metric'].__name__}@{metric['k']}:\t{score:.4f}")
