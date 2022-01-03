import os
import glob
import json
import random
from PIL import Image


root = '/home/atao/data/vball'

# number of digits possible: 0-9
NUM_DIGITS = 10


def get_images():
    """
    Assemble a list of (filename, jersey number) of all images
    """
    all_images = glob.glob(os.path.join(root, '*/jerseys/19/*.jpg'))
    all_images = glob.glob(os.path.join(root, '*/jerseys/[0-9]*/*.jpg'))
    print(f'Found {len(all_images)} images')

    data_list = []

    for filename in all_images:
        jersey_number = int(os.path.basename(os.path.dirname(filename)))
        data_list.append((filename, jersey_number))

    return data_list


def data_list_to_coco(data_list, split='train'):
    json_dict = {}
    json_dict['images'] = []
    json_dict['annotations'] = []
    json_dict['categories'] = \
        [{'id': i, 'name': str(i)} for i in range(NUM_DIGITS)]

    ann_idx = 0

    for image_id, (filename, jersey_number) in enumerate(data_list):

        # For a jersey, we create a single fake bbox who's label
        # is an integer with value = the full jersey number.
        annotation = {
            'iscrowd': 0,
            'image_id': image_id,
            'bbox': (0, 0, 0, 0),
            'area': (100),
            'category_id': jersey_number,
            'id': ann_idx,
        }
        json_dict['annotations'].append(annotation)
        ann_idx += 1

        # Convert image metadata
        img = Image.open(filename)
        w, h = img.size
        item = {
            'file_name': filename,
            'height': h,
            'width': w,
            'id': image_id
            }
        json_dict['images'].append(item)

    outfile = os.path.join(root, f'jerseys_{split}.json')
    with open(outfile, 'w') as out_fp:
        json.dump(json_dict, out_fp)
    print(f'Wrote {outfile}')


def load_json(json_fn):
    fp = open(json_fn, 'r')
    data = json.load(fp)
    return data


data_list = get_images()

# create trn/val split
num_images = len(data_list)
num_trn = int(num_images * 0.85)
random.shuffle(data_list)
trn_list = data_list[:num_trn]
val_list = data_list[num_trn:]

data_list_to_coco(trn_list, split='trn')
data_list_to_coco(val_list, split='val')
