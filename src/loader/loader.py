"""
Dataset loader
"""

import random
import matplotlib.pyplot as plt
import numpy as np

from tensorpack.dataflow import (AugmentImageComponent, AugmentImageComponents,
                                 BatchData, BatchDataByShape, CacheData,
                                 PrefetchDataZMQ, RNGDataFlow, RepeatedData)

from config import Config

####


class DatasetSerial(RNGDataFlow, Config):
    """
    Produce ``(image, label)`` pair, where
        ``image`` has shape HWC and is RGB, has values in range [0-255].

        ``label`` is a float image of shape (H, W, C). Number of C depends
                  on `self.model_mode` within `config.py`
    """

    def __init__(self, path_list):
        super(DatasetSerial, self).__init__()
        self.path_list = path_list
    ##

    def size(self):
        return len(self.path_list)
    ##

    def get_data(self):
        idx_list = list(range(0, len(self.path_list)))
        random.shuffle(idx_list)
        for idx in idx_list:

            if self.model_mode == 'seg_gland' or self.model_mode == 'seg_nuc':
                data = np.load(self.path_list[idx])
                # split stacked channel into image and label
                img = data[..., :3]  # RGB image
                img = img.astype('uint8')
                lab = data[..., 3:]  # instance ID map
                yield [img, lab]

            else:
                data = np.load(self.path_list[idx])
                # split stacked channel into image and label
                img = data[:-1]  # RGB image
                # reshape vector to HxWxC
                img = np.reshape(
                    img, self.train_input_shape + [self.input_chans])
                if self.model_mode != 'class_rotmnist':
                    img = img.astype('uint8')

                lab = data[-1]  # label
                lab = np.reshape(lab, [1, 1, 1])

                yield [img, lab]

####


def valid_generator_seg(ds, shape_aug=None, input_aug=None,
                        label_aug=None, batch_size=16, nr_procs=1):
    ### augment both the input and label
    ds = ds if shape_aug is None else AugmentImageComponents(
        ds, shape_aug, (0, 1), copy=True)
    ### augment just the input
    ds = ds if input_aug is None else AugmentImageComponent(
        ds, input_aug, index=0, copy=False)
    ### augment just the output
    ds = ds if label_aug is None else AugmentImageComponent(
        ds, label_aug, index=1, copy=True)
    #
    ds = BatchData(ds, batch_size, remainder=True)
    ds = CacheData(ds)  # cache all inference images
    return ds
####


def valid_generator_class(ds, shape_aug=None, input_aug=None,
                          batch_size=16, nr_procs=1):
    ### augment the input
    ds = ds if shape_aug is None else AugmentImageComponent(
        ds, shape_aug, index=0, copy=True)
    ### augment the input
    ds = ds if input_aug is None else AugmentImageComponent(
        ds, input_aug, index=0, copy=False)
    #
    ds = BatchData(ds, batch_size, remainder=True)
    ds = CacheData(ds)  # cache all inference images
    return ds

####


def train_generator_seg(ds, shape_aug=None, input_aug=None,
                        label_aug=None, batch_size=16, nr_procs=8):
    ### augment both the input and label
    ds = ds if shape_aug is None else AugmentImageComponents(
        ds, shape_aug, (0, 1), copy=True)
    ### augment just the input i.e index 0 within each yield of DatasetSerial
    ds = ds if input_aug is None else AugmentImageComponent(
        ds, input_aug, index=0, copy=False)
    ### augment just the output i.e index 1 within each yield of DatasetSerial
    ds = ds if label_aug is None else AugmentImageComponent(
        ds, label_aug, index=1, copy=True)
    #
    ds = BatchDataByShape(ds, batch_size, idx=0)
    ds = PrefetchDataZMQ(ds, nr_procs)
    return ds
####


def train_generator_class(ds, shape_aug=None, input_aug=None, batch_size=16, nr_procs=8):
    ### augment the input
    ds = ds if shape_aug is None else AugmentImageComponent(
        ds, shape_aug, index=0, copy=True)
    ### augment the input i.e index 0 within each yield of DatasetSerial
    ds = ds if input_aug is None else AugmentImageComponent(
        ds, input_aug, index=0, copy=False)
    #
    ds = BatchDataByShape(ds, batch_size, idx=0)
    ds = PrefetchDataZMQ(ds, nr_procs)
    return ds

####


def visualize(datagen, batch_size):
    """
    Read the batch from 'datagen' and display 'view_size' number of
    of images and their corresponding Ground Truth
    """
    cfg = Config()

    def prep_imgs(img, lab):

        # Deal with HxWx1 case
        img = np.squeeze(img)

        if cfg.model_mode == 'seg_gland' or cfg.model_mode == 'seg_nuc':
            cmap = plt.get_cmap('jet')
            # cmap may randomly fails if of other types
            lab = lab.astype('float32')
            lab_chs = np.dsplit(lab, lab.shape[-1])
            for i, ch in enumerate(lab_chs):
                ch = np.squeeze(ch)
                # cmap may behave stupidly
                ch = ch / (np.max(ch) - np.min(ch) + 1.0e-16)
                # take RGB from RGBA heat map
                lab_chs[i] = cmap(ch)[..., :3]
            img = img.astype('float32') / 255.0
            prepped_img = np.concatenate([img] + lab_chs, axis=1)
        else:
            prepped_img = img
        return prepped_img

    ds = RepeatedData(datagen, -1)
    ds.reset_state()
    for imgs, labs in ds.get_data():
        if cfg.model_mode == 'seg_gland' or cfg.model_mode == 'seg_nuc':
            for idx in range(0, 4):
                displayed_img = prep_imgs(imgs[idx], labs[idx])
                # plot the image and the label
                plt.subplot(4, 1, idx+1)
                plt.imshow(displayed_img, vmin=-1, vmax=1)
                plt.axis('off')
            plt.show()
        else:
            for idx in range(0, 8):
                displayed_img = prep_imgs(imgs[idx], labs[idx])
                # plot the image and the label
                plt.subplot(2, 4, idx+1)
                plt.imshow(displayed_img)
                if len(cfg.label_names) > 0:
                    lab_title = cfg.label_names[int(labs[idx])]
                else:
                    lab_tite = int(labs[idx])
                plt.title(lab_title)
                plt.axis('off')
            plt.show()
    return
###

###########################################################################
