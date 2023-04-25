import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename1', type=str, default="performance_of_1.json")
parser.add_argument('--filename2', type=str, default="performance_of_2.json")
args = parser.parse_args()

with open(args.filename1, 'r') as file_obj1:
    data1 = json.load(file_obj1)
with open(args.filename2, 'r') as file_obj2:
    data2 = json.load(file_obj2)

infer_time = (data1['infer_time'] + data2['infer_time'])/2
RTF = (data1['RTF'] + data2['RTF'])/2
print('**************Averange Performance**************')
print(f'averange infer time: {infer_time} s')
print(f'averange RTF: {RTF}')