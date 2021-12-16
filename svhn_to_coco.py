import os
import h5py
import json
import random

from PIL import Image
from tqdm import tqdm


root = '/home/atao/data/SVHN'


def get_img_name(f, idx=0):
    names = f['digitStruct/name']
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return(img_name)


def get_img_boxes(f, idx=0):
    """
    Get the 'height', 'left', 'top', 'width', 'label' of bounding boxes
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary
    """
    bbox_prop = ['height', 'left', 'top', 'width', 'label']
    meta = {key: [] for key in bbox_prop}
    bboxs = f['digitStruct/bbox']

    box = f[bboxs[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))

    return meta


def convert(f, split):
    names = f['digitStruct/name']
    num_images = len(names)
    json_dict = {}
    json_dict['images'] = []
    json_dict['annotations'] = []
    json_dict['categories'] = [{'id': i, 'name': str(i)} for i in range(10)]
    ann_idx = 0

    for image_id in tqdm(range(num_images)):
        # Convert bbox metadata
        img_boxes = get_img_boxes(f, image_id)
        labels = img_boxes['label']

        num_digits = len(labels)

        # Only use images with 1 or 2 numbers ...
        # if num_digits > 2:
        #    continue

        bboxes = []
        labels = []
        for j in range(num_digits):
            bboxes.append([img_boxes['left'][j], img_boxes['top'][j],
                           img_boxes['width'][j], img_boxes['height'][j]])
            labels.append(img_boxes['label'][j])

        for bbox, label in zip(bboxes, labels):
            annotation = {
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'category_id': label,
                'id': ann_idx,
            }
            json_dict['annotations'].append(annotation)
            ann_idx += 1

        # Convert image metadata
        filename = get_img_name(f, image_id)
        filename = os.path.join(root, split, filename)
        img = Image.open(filename)
        w, h = img.size
        item = {
            'file_name': filename,
            'height': h,
            'width': w,
            'id': image_id
            }
        json_dict['images'].append(item)

    num_annotations = len(json_dict['images'])
    print(f'Found {num_annotations} images with 1 or 2 numbers')

    outfile = os.path.join(root, f'{split}.json')
    with open(outfile, 'w') as outfile:
        json.dump(json_dict, outfile)


def convert_splits():
    for split in ['train', 'extra']:
        f = h5py.File(f'{split}.mat', 'r')
        convert(f, split)


def load_json(json_fn):
    fp = open(json_fn, 'r')
    data = json.load(fp)
    return data


def fuse_and_split(val_pct=0.15):
    # make the val dataset out of 15% of train
    # use the remainder of train + all of extra to make the joint_train split
    train_data = load_json(os.path.join(root, 'train.json'))
    extra_data = load_json(os.path.join(root, 'extra.json'))

    images = train_data['images']
    random.shuffle(images)

    num_total = len(images)
    num_anns = len(train_data['annotations'])
    num_val = int(val_pct * num_total)

    val_images = images[:num_val]
    trn_images = images[num_val:]

    val_ids = [img['id'] for img in val_images]

    trn_annotations = []
    val_annotations = []

    for ann in train_data['annotations']:
        if ann['image_id'] in val_ids:
            val_annotations.append(ann)
        else:
            trn_annotations.append(ann)

    trn_json = {}
    val_json = {}

    trn_json['categories'] = train_data['categories']
    val_json['categories'] = train_data['categories']

    trn_json['images'] = trn_images
    val_json['images'] = val_images

    trn_json['annotations'] = trn_annotations
    val_json['annotations'] = val_annotations

    # process extra data:
    for img_dict in extra_data['images']:
        img_dict['id'] += num_total
    for ann in extra_data['annotations']:
        ann['image_id'] += num_total
        ann['id'] += num_anns

    trn_json['images'].extend(extra_data['images'])
    trn_json['annotations'].extend(extra_data['annotations'])

    total_trn = len(trn_json['images'])
    print(f'{total_trn} total training images')

    with open(os.path.join(root, 'trn_split.json'), 'w') as fp:
        json.dump(trn_json, fp)
        fp.close()

    with open(os.path.join(root, 'val_split.json'), 'w') as fp:
        json.dump(val_json, fp)
        fp.close()


convert_splits()
fuse_and_split()
