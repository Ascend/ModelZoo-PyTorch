import time
import os
import argparse
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--data_path', default='',metavar='DIR',
                    help='path to dataset')
args = parser.parse_args()

with open("train_record","w")as f:
    f.write('0')
os.system("bash test/train_full_8p_0-120.sh "+"--data_path="+args.data_path)
while True:
    with open("train_record","r")as f:
        sig=f.read()
    if sig=="0":
        pass
    else:
        os.system("bash test/train_full_8p_121-300.sh "+"--data_path="+args.data_path)
        break

    time.sleep(600)
