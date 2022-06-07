import re
import sys
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, default="./msame_bs1.txt")
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    if args.result_file.endswith('.json'):
        result_json = args.result_file
        with open(result_json, 'r') as f:
            content = f.read()
        tops = [i.get('value') for i in json.loads(content).get('value') if 'Top' in i.get('key')]
        print('om {} top1:{}'.format(result_json.split('_')[1].split('.')[0], tops[0]))
    elif args.result_file.endswith('.txt'):
        result_txt = args.result_file
        with open(result_txt, 'r') as f:
            content = f.read()
        txt_data_list = re.findall(r'Inference average time without first time:.*ms', content.replace('\n', ',') + ',')[-1]
        avg_time = txt_data_list.split(' ')[-2]
        fps = args.batch_size * 1000 / float(avg_time)
        print('310P bs{} fps:{:.3f}'.format(args.batch_size, fps))