"""extract_patches.py

Script for extracting patches from image tiles. The script will read
and RGB image and a corresponding label and form image patches to be
used by the network.
"""


import glob
import os

import cv2
import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from config import Config

###########################################################################
if __name__ == "__main__":

    cfg = Config()

    extract_type = "mirror"  # 'valid' or 'mirror'
    # 'mirror' reflects at the borders; 'valid' doesn't.
    # check the patch_extractor.py 'main' to see the difference

    # original size (win size) - input size - output size (step size)
    step_size = [112, 112]
    # set to size of network input: 448 for glands, 256 for nuclei
    win_size = [448, 448]

    xtractor = PatchExtractor(win_size, step_size)

    ### Paths to data - these need to be modified according to where the original data is stored
    img_ext = ".png"
    # img_dir should contain RGB image tiles from where to extract patches.
    img_dir = "path/to/images/"
    # ann_dir should contain 2D npy image tiles, with values ranging from 0 to N.
    # 0 is background and then each nucleus is uniquely labelled from 1-N.
    ann_dir = "path/to/labels/"
    ####
    out_dir = "output_path/%dx%d_%dx%d" % (
        win_size[0],
        win_size[1],
        step_size[0],
        step_size[1],
    )

    file_list = glob.glob("%s/*%s" % (img_dir, img_ext))
    file_list.sort()

    rm_n_mkdir(out_dir)
    for filename in file_list:
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]
        print(filename)

        img = cv2.imread(img_dir + basename + img_ext)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # assumes that ann is HxW
        ann_inst = np.load(ann_dir + basename + ".npy")
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)

        img = np.concatenate([img, ann], axis=-1)
        sub_patches = xtractor.extract(img, extract_type)
        for idx, patch in enumerate(sub_patches):
            np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)
