import os
import sys
from glob import glob

file_path = sys.argv[1]
info_name = sys.argv[2]
width = sys.argv[3]
height = sys.argv[4]

bin_images = glob(os.path.join(file_path, '*'))

with open(info_name, 'w') as  file:
    for index, img in enumerate(bin_images):
        content = ' '.join([str(index), img, width, height])
        file.write(content)
        file.write('\n')
