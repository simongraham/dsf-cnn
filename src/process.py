"""
Post-processing
"""

import glob
import os
import time
import cv2
import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects, watershed

from config import Config

from misc.viz_utils import visualize_instances
from misc.utils import remap_label


def process_utils(pred_map, mode):
    """
    Performs post processing for a given image

    Args:
        pred_map: output of CNN
        mode: choose either 'seg_gland' or 'seg_nuc'
    """

    if mode == "seg_gland":
        pred = np.squeeze(pred_map)

        blb = pred[..., 0]
        blb = np.squeeze(blb)
        cnt = pred[..., 1]
        cnt = np.squeeze(cnt)
        cnt[cnt > 0.5] = 1
        cnt[cnt <= 0.5] = 0

        pred = blb - cnt
        pred[pred > 0.55] = 1
        pred[pred <= 0.55] = 0
        k_disk1 = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            np.uint8,
        )
        # ! refactor these
        pred = binary_fill_holes(pred)
        pred = pred.astype("uint16")
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, k_disk1)
        pred = measurements.label(pred)[0]
        pred = remove_small_objects(pred, min_size=1500)

        k_disk2 = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
            ],
            np.uint8,
        )

        pred = pred.astype("uint16")
        proced_pred = cv2.dilate(pred, k_disk2, iterations=1)
    elif mode == "seg_nuc":
        blb_raw = pred_map[..., 0]
        blb_raw = np.squeeze(blb_raw)
        blb = blb_raw.copy()
        blb[blb > 0.5] = 1
        blb[blb <= 0.5] = 0
        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1

        mrk_raw = pred_map[..., 1]
        mrk_raw = np.squeeze(mrk_raw)
        cnt_raw = pred_map[..., 2]
        cnt_raw = np.squeeze(cnt_raw)
        cnt = cnt_raw.copy()
        cnt[cnt >= 0.4] = 1
        cnt[cnt < 0.4] = 0
        mrk = mrk_raw - cnt
        mrk = mrk * blb
        mrk[mrk > 0.75] = 1
        mrk[mrk <= 0.75] = 0

        marker = mrk.copy()
        marker = binary_fill_holes(marker)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=10)
        proced_pred = watershed(-mrk_raw, marker, mask=blb)

    return proced_pred


def process():
    """
    Performs post processing for a list of images

    """

    cfg = Config()

    for data_dir in cfg.inf_data_list:

        proc_dir = cfg.inf_output_dir + "/processed/"
        pred_dir = cfg.inf_output_dir + "/raw/"
        file_list = glob.glob(pred_dir + "*.npy")
        file_list.sort()  # ensure same order

        if not os.path.isdir(proc_dir):
            os.makedirs(proc_dir)
        for filename in file_list:
            start = time.time()
            filename = os.path.basename(filename)
            basename = filename.split(".")[0]

            test_set = basename.split("_")[0]
            test_set = test_set[-1]

            print(pred_dir, basename, end=" ", flush=True)

            ##
            img = cv2.imread(data_dir + basename + cfg.inf_imgs_ext)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred_map = np.load(pred_dir + "/%s.npy" % basename)

            # get the instance level prediction
            pred_inst = process_utils(pred_map, cfg.model_mode)

            # ! remap label is slow - check to see whether it is needed!
            pred_inst = remap_label(pred_inst, by_size=True)

            overlaid_output = visualize_instances(pred_inst, img)
            overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
            cv2.imwrite("%s/%s.png" % (proc_dir, basename), overlaid_output)

            # save segmentation mask
            np.save("%s/%s" % (proc_dir, basename), pred_inst)

            end = time.time()
            diff = str(round(end - start, 2))
            print("FINISH. TIME: %s" % diff)


if __name__ == "__main__":
    process()

