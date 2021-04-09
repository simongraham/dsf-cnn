"""
Visualisation utils
"""

import math
import random
import colorsys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .utils import bounding_box


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def visualize_instances(mask, canvas=None):
    """
    Args:
        mask: array of NW
    Return:
        Image with the instance overlaid
    """

    colour = [255, 255, 0]  # yellow

    canvas = (
        np.full((mask.shape[0], mask.shape[1]) + (3,), 200, dtype=np.uint8)
        if canvas is None
        else np.copy(canvas)
    )

    insts_list = list(np.unique(mask))
    insts_list.remove(0)  # remove background

    for idx, inst_id in enumerate(insts_list):
        inst_map = np.array(mask == inst_id, np.uint8)
        y1, y2, x1, x2 = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_canvas_crop = canvas[y1:y2, x1:x2]
        contours = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(inst_canvas_crop, contours[0], -1, colour, 3)
        canvas[y1:y2, x1:x2] = inst_canvas_crop

    return canvas
