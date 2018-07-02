"""
    This code for generating the training/testing split strategy based on paper:

    "AlignedReID: Surpassing Human-Level Performance in Person Re-Identification".

    For training: 1267 ids
    For test: 200 ids
        query: 400 images

"""

from __future__ import print_function

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import numpy as np

from aligned_reid.utils.utils import save_pickle
from aligned_reid.utils.dataset_utils import parse_im_name

total_id_num = 1467
train_id_num = 1267
test_id_num = 200

cam_num = 2


def split_train_test_set(img_dir, save_pkl_dir):
    train_im_names = []
    test_im_names = []
    gallery_im_names = []
    query_im_names = []
    query_ids = []

    file_list = os.listdir(img_dir)
    file_list.sort()

    test_id = random.sample(range(total_id_num), test_id_num)
    test_id.sort()

    # get the id of training/testing in the form of 'XXXX'
    test_ids = ['{0:08}'.format(i) for i in test_id]
    train_ids = ['{0:08}'.format(i) for i in range(total_id_num) if i not in test_id]
    cam_ids = ['{0:04}'.format(i) for i in range(cam_num)]

    # generate training set and test set
    for tr in train_ids:
        for f in file_list:
            if tr == '{0:08}'.format(parse_im_name(f, 'id')):
                train_im_names.append(f)

    for te in test_ids:
        for f in file_list:
            if te == '{0:08}'.format(parse_im_name(f, 'id')):
                test_im_names.append(f)

    # generate query in test set
    item = 0
    while item < test_id_num:
        i_1, i_2 = random.sample(test_im_names, 2)
        if parse_im_name(i_1, 'id') == parse_im_name(i_2, 'id')\
                and parse_im_name(i_1, 'id') not in query_ids:
            if parse_im_name(i_1, 'cam') != parse_im_name(i_2, 'cam'):
                item += 1
                query_im_names.append(i_1)
                query_im_names.append(i_2)
                query_ids.append(parse_im_name(i_1, 'id'))
            else:
                continue
        else:
            continue

    # generate gallery in test set
    gallery_im_names = [f for f in test_im_names if f not in query_im_names]

    train_im_names.sort()
    gallery_im_names.sort()
    query_im_names.sort()

    partitions = {'gallery_im_names': np.asarray(gallery_im_names),
                  'query_im_names': np.asarray(query_im_names),
                  'train_im_names': np.asarray(train_im_names)}
    partitions = {'detected': partitions,
                  'labeled': partitions}
    save_pickle(partitions, save_pkl_dir)

    print("Spliting operation has been done!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Split CUHK03 Dataset to Train/Test Set")
    parser.add_argument(
        '--imgs_file',
        type=str,
        default='/database/reid_dataset/cuhk03_github/detected/images')
    parser.add_argument(
        '--train_test_partition_file',
        type=str,
        default='/database/reid_dataset/cuhk03_github/paper_train_test_split.pkl')
    args = parser.parse_args()
    imgs_file = osp.abspath(osp.expanduser(args.imgs_file))
    train_test_partition_file = osp.abspath(osp.expanduser(args.train_test_partition_file))
    split_train_test_set(imgs_file, train_test_partition_file)
