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
        img = img[h_0:h_0 + crop_shape[0], w_0:w_0 + crop_shape[1]]
    else:
        h_0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w_0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        img = img[:, h_0:h_0 + crop_shape[0], w_0:w_0 + crop_shape[1]]
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
        files = glob.glob(sub_dir + '/*' + data_ext)
        data_files.extend(files)

    return data_files
