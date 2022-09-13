import json

with open('./instances_valminusminival2014.json', 'r') as file:
    content = file.read()
content = json.loads(content)
info = content.get('info')
licenses = content.get('licenses')
images = content.get('images')
annotations = content.get('annotations')
categroies = content.get('categories')

with open('./coco2014.names', 'w') as f:
    for categroie in categroies:
        f.write(categroie.get('name'))
        f.write('\n')

file_names = [image.get('file_name') for image in images]
widths = [image.get('width') for image in images]
heights = [image.get('height') for image in images]
image_ids = [image.get('id') for image in images]
assert len(file_names) == len(widths) == len(heights) == len(image_ids), "must be equal"

annotation_ids = [annotation.get('image_id') for annotation in annotations]
bboxs = [annotation.get('bbox') for annotation in annotations]
category_ids = [annotation.get('category_id') for annotation in annotations]
segmentations = [annotation.get('segmentation') for annotation in annotations]
iscrowds = [annotation.get('iscrowd') for annotation in annotations]

assert len(annotation_ids) == len(bboxs) == len(category_ids) ==len(segmentations) # 255094

with open('coco_2014.info', 'w') as f:
    for index, file_name in enumerate(file_names):
        file_name = 'val2014/' + file_name
        line = "{} {} {} {}".format(index, file_name, widths[index], heights[index])
        f.write(line)
        f.write('\n')

def get_all_index(lst, item):
    return [index for (index, value) in enumerate(lst) if value == item]

def get_categroie_name(lst, item):
    categroie_name =  [dt.get('name') for dt in lst if item == dt.get('id')][0]
    if len(categroie_name.split()) == 2:
        temp = categroie_name.split()
        categroie_name = temp[0] + '_' + temp[1]
    return categroie_name

for index, image_id in enumerate(image_ids):
    indexs = get_all_index(annotation_ids, image_id)
    with open('./ground-truth-split/{}.txt'.format(file_names[index].split('.')[0]), 'w') as f:
        for idx in indexs:
            f.write(get_categroie_name(categroies, category_ids[idx]))
            print(get_categroie_name(categroies, category_ids[idx]))
            f.write(' ')
            # change label
            bboxs[idx][2] = bboxs[idx][0] + bboxs[idx][2]
            bboxs[idx][3] = bboxs[idx][1] + bboxs[idx][3]
            f.write(' '.join(map(str, bboxs[idx])))
            f.write('\n')




