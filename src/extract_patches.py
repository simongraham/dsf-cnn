
import glob
import os
import cv2
import numpy as np
from scipy.ndimage.measurements import label
import math

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

###########################################################################
if __name__ == '__main__':

    extract_type = 'mirror' # 'valid' extract patches without reflection at boundary.
                            # 'mirror' extract patches with mirror padding at boundary.

    # original size (win size) - input size - output size (step size)
    step_size = [112, 112]
    win_size  = [512, 512]

    xtractor = PatchExtractor(win_size, step_size)

    ##
    img_ext = '.tif'
    ann_ext = '.npy'

    img_dir = '/media/simon/Storage 1/Data/Nuclei/kumar_consep/kumar/train/Images/'
    ann_dir = '/media/simon/Storage 1/Data/Nuclei/kumar_consep/kumar/train/Labels/' 
    if use_tissue_mask:
        tissue_dir = '/media/simon/Storage 1/Data/Colon/digestpath/Colonoscopy_tissue_segment_dataset/Labels_temp/Tissue/' 
    ####
    out_dir = "/media/simon/Storage 1/Data/Nuclei/patches/kumar2/train/"

    file_list = glob.glob('%s/*%s' % (img_dir, img_ext))

    rm_n_mkdir(out_dir)

    for filename in file_list:
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]
        print(filename)

        img = cv2.imread(img_dir + basename + img_ext)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = img.astype('uint8')

        ann = np.load(ann_dir + basename + '.npy')
        ann = ann.astype('int32')

        ann = ann.astype('int32')
        ann = np.expand_dims(ann, -1)

        img = np.concatenate([img, ann], axis=-1)
        sub_patches = xtractor.extract(img, extract_type)
        for idx, patch in enumerate(sub_patches):
            np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)

