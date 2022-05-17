from glob import glob
import json
import sys
from ECAPA-TDNN.mel2samp_tacotron2 import Mel2SampWaveglow
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
import librosa
import csv
import os

from functools import partial
from pydub import AudioSegment
from tqdm import tqdm
from queue import Queue
import threading

from itertools import chain

CONFIGURATION_FILE = 'config.json'
T_THRES = 19
DATA_SET = sys.argv[1]

with open(CONFIGURATION_FILE) as f:
    data = f.read()
    json_info = json.loads(data)

    mel_config = json_info["mel_config"]
    MEL2SAMPWAVEGLOW = Mel2SampWaveglow(**mel_config)

    hp = json_info["hp"]

    global_scope = sys.modules[__name__]

    for key in hp:
        setattr(global_scope, key, hp[key])
        # print(f'{key} == {hp[key]}')

def struct_meta(file_list, mode='vox1'):
    if mode == 'vox1':
        meta = [(file, file.split('/')[2],
                librosa.get_duration(filename=file), 
                librosa.get_samplerate(file)
                ) for file in tqdm(file_list)]
    elif mode == 'vox2':

        def meta_thread(file_list, queue):
            def get_duration_and_sample_rate(file):
                audio = AudioSegment.from_file(file)
                duration = audio.frame_count() / audio.frame_rate
                return duration, audio.frame_rate

            meta = [(file, file.split('/')[-3],
                    *get_duration_and_sample_rate(file)
                    ) for file in tqdm(file_list)]

            queue.put(meta)
            
            return
        
        num_thread = 8
        thread_queue = Queue()
        threads = [threading.Thread(target=meta_thread, args=(file_list[i::num_thread], thread_queue,)) for i in range(num_thread)]
        for t in threads: t.start()
        for t in threads: t.join() 
            
        meta = list(chain(*[thread_queue.get() for i in range(num_thread)]))

    else:
        assert False, f'Unknown mode {mode}'
    return meta


def reduce_meta(meta, speaker_num=100):

    if speaker_num < 0:
        return meta

    from collections import Counter
    c = Counter()

    for m in meta:
        c[m[1]] += 1

    top_n = [d[0] for d in c.most_common(speaker_num)]

    meta_filtered = list(filter(lambda x: x[1] in top_n, meta))
    
    return meta_filtered


def mel_random_masking(tensor, masking_ratio=0.1, mel_min=-12):

    mask = torch.rand(tensor.shape) > masking_ratio

    masked_tensor = torch.mul(tensor, mask)

    masked_tensor += ~mask * mel_min

    return masked_tensor

def apply_t_shift(tensor, mel_min=-12, T=10):
    
    _, MF = tensor.shape
    
    t = torch.randint(0, T, [1])
    
    shift_tensor = torch.ones(t, MF) * mel_min

    tensor = torch.cat((shift_tensor, tensor), axis=0)
    
    return tensor

def normalize_tensor(tensor, min_v=-12, max_v=0):
    center_v = (max_v - min_v) / 2
    tensor = tensor / center_v  + 1
    return tensor
    
def plot_mel_spectrograms(mel_tensor, keyword=''):

    B, M, T = mel_tensor.shape

    num_x = int(np.sqrt(B))
    num_y = int(B / num_x)

    fig, axes = plt.subplots(num_x, num_y, sharex=True, sharey=True, figsize=(24, 8), dpi=300)
    axes = axes.flatten()

    for i in range(B):
        im = axes[i].imshow(mel_tensor[i, :, :], origin='lower', aspect='auto')

    plt.tight_layout()

    fig.subplots_adjust(right=0.94)
    cbar_ax = fig.add_axes([0.96, 0.05, 0.02, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.savefig(f'mel_sample_{keyword}.png')
    plt.close()

    return

def pick_random_mel_segments(mel, max_mel_length):
    L, _ = mel.shape
    offset =  np.random.randint(L - max_mel_length)
    mel = mel[offset:offset+max_mel_length, :]
    return mel

def wav2mel_tensor(wav_files):

    B = len(wav_files)
    mels = list()

    for wav_file in wav_files:
        mel = MEL2SAMPWAVEGLOW.get_mel(wav_file).T # (MB, T) -> (T, MB)
        # mel = pick_random_mel_segments(mel, max_mel_length)

        mel = normalize_tensor(mel, MEL_MIN)
        mels.append(mel) 

    mel_tensor = pad_sequence(mels, batch_first=True, padding_value=-1).transpose(1, 2) # (B, T, MB) -> (B, MB, T)

    return mel_tensor


def collate_function(pairs, speaker_table, max_mel_length=-1):

    mels = list()
    speakers = list()
    mel_lengths = list()

    B = len(pairs)

    for pair in pairs:
        # (wav_file, clean_script, clean_jamos, tag, len(clean_script), len(clean_jamos), wav_file_dur)
        wav_file = pair[0]
        speaker =  pair[1]
        npy_file = wav_file.replace('.wav', '.npy')
        if not os.path.isfile(npy_file) or not LOAD_MEL_FROM_DISK:
        # if True:
            mel = MEL2SAMPWAVEGLOW.get_mel(wav_file).T # (MB, T) -> (T, MB)
            np.save(npy_file, mel)
        else:
            mel = torch.tensor(np.load(npy_file)) # (T, MB)
        mel = pick_random_mel_segments(mel, max_mel_length)
        # mel = mel[:max_mel_length, :]
        # Pick random mel

        if APPLY_T_SHIFT:
            mel = apply_t_shift(mel, MEL_MIN)
        mel = mel_random_masking(mel, MASKING_RATIO, MEL_MIN)
        mel = normalize_tensor(mel, MEL_MIN)
        mels.append(mel) 
        mel_lengths.append(mel.shape[0])
        speakers.append(speaker_table[speaker])

    mel_tensor = pad_sequence(mels, batch_first=True, padding_value=-1).transpose(1, 2) # (B, T, MB) -> (B, MB, T)
    mel_lengths = torch.tensor(mel_lengths)
    speakers = torch.tensor(speakers)

    return mel_tensor, mel_lengths, speakers

def write_to_csv(meta_data, file_name):
    with open(f'{file_name}', 'w') as f:
        csv_writer = csv.writer(f)
        for meta in meta_data:
            csv_writer.writerow(meta)
    
    return

def read_from_csv(file_name):
    with open(f'{file_name}', 'r') as f:
        csv_reader = csv.reader(f)
        meta = [(line[0], line[1], float(line[2]), float(line[3])) for line in tqdm(csv_reader)]
             
    return meta

class SpeakerDict():

    def __init__(self, speakers):
        self.speaker_array = sorted(speakers)
        self.speaker_dict = {s: i for i, s in enumerate(self.speaker_array)}

    def __getitem__(self, key):

        if isinstance(key, int):
            return self.speaker_array[key]

        elif isinstance(key, str):
            return self.speaker_dict[key]

        else:
            assert False, f'Invalid key for SpeakerDict {key}'

    def __len__(self):
        a = len(self.speaker_array)
        b = len(self.speaker_dict)
        assert a == b, f'{a} != {b}'
        return a

    def decode_speaker_tensor(self, tensor):
        return [self.speaker_array[v] for v in tensor]

def build_speaker_dict(meta):

    speakers = list(set([m[1] for m in meta]))
    speaker_dict =  SpeakerDict(speakers)

    return speaker_dict

def load_meta(dataset, keyword='vox1'):

    if keyword == 'vox1':

        wav_files_test = sorted(glob(dataset +'/vox1_test' + '/*/*/*.wav'))
        print(f'Len. wav_files_test {len(wav_files_test)}')

        if not os.path.isfile('vox1_test.csv'):
            test_meta = struct_meta(wav_files_test)
            write_to_csv(test_meta, 'vox1_test.csv')
        else:
            test_meta = read_from_csv('vox1_test.csv')
    
    return  test_meta

def get_dataloader(keyword='vox1', t_thres=19,dataset = DATA_SET):
    test_meta = load_meta(dataset, keyword)
    
    test_meta_ = [meta for meta in tqdm(test_meta) if meta[2] < t_thres]
   


    test_meta = reduce_meta(test_meta_, speaker_num=REDUCED_SPEAKER_NUM_TEST)
    print(f'Meta reduced {len(test_meta_)} => {len(test_meta)}')
    
    test_speakers = build_speaker_dict(test_meta)
    

    
    # dataset_test = DataLoader(test_meta, batch_size=BATCH_SIZE, 
    #                           shuffle=False, num_workers=NUM_WORKERS,
    #                           collate_fn=lambda x: collate_function(x, test_speakers),
    #                           drop_last=True)
    # AttributeError: Can't pickle local object 'get_dataloader.<locals>.<lambda>'
    # AttributeError: Can't pickle local object 'get_dataloader.<locals>.dev_collator'

    # dataset_test = DataLoader(test_meta, batch_size=BATCH_SIZE, 
    #                         shuffle=False, num_workers=NUM_WORKERS,
    #                         collate_fn=test_collator,
    #                         drop_last=True)

    # dataset_test = DataLoader(test_meta, batch_size=BATCH_SIZE, 
    dataset_test = DataLoader(test_meta, batch_size=16,
                              shuffle=False, num_workers=2,
                              collate_fn=partial(collate_function, 
                                                 speaker_table=test_speakers,
                                                 max_mel_length=MAX_MEL_LENGTH),
                              prefetch_factor=2,
                              pin_memory=True,
                              drop_last=True)

    return dataset_test, test_speakers

if __name__ == "__main__":
    predata_path = sys.argv[2]
    prespeaker_path = sys.argv[3]
    dataset_test, test_speakers = get_dataloader('vox1', 19)
    if not os.path.exists(predata_path):  #判断是否存在文件夹如果不存在则创建为文件夹
       os.makedirs(predata_path)
    if not os.path.exists(prespeaker_path):  
       os.makedirs(prespeaker_path)
    i=0
    for mels, mel_length, speakers in tqdm(dataset_test):
      i=i+1
      mels = np.array(mels).astype(np.float32)
      mels.tofile(predata_path+'mels'+str(i)+".bin")
      #speakers = np.array(speakers).astype(np.int64)
      #speakers.tofile('prespeaker/speakers'+str(i)+".txt")
      #np.savetxt('prespeaker/speakers'+str(i)+".txt",speakers)
      torch.save(speakers,prespeaker_path + 'speakers'+str(i)+".pt")
      
        


