"""
Utils
"""

import glob
import os
import shutil
import cv2
import numpy as np


def bounding_box(img):
    """
    Get the bounding box of a binary region

    Args:
        img: input array- should contain one
             binary object.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


####


def cropping_center(img, crop_shape, batch=False):
    """
    Crop an array at the centre

    Args:
        img: input array
        crop_shape: new spatial dimensions (h,w)
    """

    orig_shape = img.shape
    if not batch:
        h_0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w_0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        img = img[h_0 : h_0 + crop_shape[0], w_0 : w_0 + crop_shape[1]]
    else:
        h_0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w_0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        img = img[:, h_0 : h_0 + crop_shape[0], w_0 : w_0 + crop_shape[1]]
    return img


####


def rm_n_mkdir(dir_path):
    """
    Remove, then create a new directory
    """

    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


####


def get_files(data_dir_list, data_ext):
    """
    Given a list of directories containing data with extention 'date_ext',
    generate a list of paths for all files within these directories
    """

    data_files = []
    for sub_dir in data_dir_list:
        files = glob.glob(sub_dir + "/*" + data_ext)
        data_files.extend(files)

    return data_files


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger object has smaller ID.
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger object has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred
