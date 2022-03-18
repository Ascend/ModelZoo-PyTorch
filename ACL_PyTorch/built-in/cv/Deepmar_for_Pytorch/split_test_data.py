import pickle
import numpy as np
import json


image = []
label = []
dataset = pickle.load(open('./dataset/peta/peta_dataset.pkl', 'rb'), encoding='utf-8')
train_split = pickle.load(open('./dataset/peta/peta_partition.pkl', 'rb'), encoding='utf-8')

for idx in train_split['test'][0]:
    image.append(dataset['image'][idx])
    label_tmp = np.array(dataset['att'][idx])[dataset['selected_attribute']].tolist()
    label.append(label_tmp)

with open('image.txt', 'w') as f:
    for name in image:
        f.write(name)
        f.write('\n')

image_label = dict(zip(image, label))
with open('label.json', 'w') as json_file:
    json.dump(image_label, json_file)


