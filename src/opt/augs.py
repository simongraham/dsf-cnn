"""
Augmentation pipeline
"""

from tensorpack import imgaug
import cv2
from loader.custom_augs import (BinarizeLabel, GaussianBlur, MedianBlur,
                                GenInstanceContourMap, GenInstanceMarkerMap)

# refer to https://tensorpack.readthedocs.io/modules/dataflow.imgaug.html for
# information on how to modify the augmentation parameters


def get_train_augmentors(self, input_shape, output_shape, view=False):
    print(input_shape, output_shape)
    if self.model_mode == 'class_rotmnist':
        shape_augs = [
            imgaug.Affine(
                rotate_max_deg=359,
                interp=cv2.INTER_NEAREST,
                border=cv2.BORDER_CONSTANT),
        ]

        input_augs = [
        ]

    else:
        shape_augs = [
            imgaug.Affine(
                rotate_max_deg=359,
                translate_frac=(0.01, 0.01),
                interp=cv2.INTER_NEAREST,
                border=cv2.BORDER_REFLECT),
            imgaug.Flip(vert=True),
            imgaug.Flip(horiz=True),
            imgaug.CenterCrop(input_shape),
        ]

        input_augs = [
            imgaug.RandomApplyAug(
                imgaug.RandomChooseAug(
                    [
                        GaussianBlur(),
                        MedianBlur(),
                        imgaug.GaussianNoise(),
                    ]
                ), 0.5),
            # Standard colour augmentation
            imgaug.RandomOrderAug(
                [imgaug.Hue((-8, 8), rgb=True),
                 imgaug.Saturation(0.2, rgb=True),
                 imgaug.Brightness(26, clip=True),
                 imgaug.Contrast((0.75, 1.25), clip=True),
                 ]),
            imgaug.ToUint8(),
        ]

    if self.model_mode == 'seg_gland':
        label_augs = []
        label_augs = [GenInstanceContourMap(mode=self.model_mode)]
        label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))
        else:
            label_augs.append(imgaug.CenterCrop(input_shape))

        return shape_augs, input_augs, label_augs
    elif self.model_mode == 'seg_nuc':
        label_augs = []
        label_augs = [GenInstanceMarkerMap()]
        label_augs.append(BinarizeLabel())
        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))
        else:
            label_augs.append(imgaug.CenterCrop(input_shape))

        return shape_augs, input_augs, label_augs

    else:
        return shape_augs, input_augs


def get_valid_augmentors(self, input_shape, output_shape, view=False):
    print(input_shape, output_shape)
    shape_augs = [
        imgaug.CenterCrop(input_shape),
    ]
    input_augs = []

    if self.model_mode == 'seg_gland':
        label_augs = []
        label_augs = [GenInstanceContourMap(mode=self.model_mode)]
        label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))
        else:
            label_augs.append(imgaug.CenterCrop(input_shape))

        return shape_augs, input_augs, label_augs
    elif self.model_mode == 'seg_nuc':
        label_augs = []
        label_augs = [GenInstanceMarkerMap()]
        label_augs.append(BinarizeLabel())
        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))
        else:
            label_augs.append(imgaug.CenterCrop(input_shape))

        return shape_augs, input_augs, label_augs
    else:
        return shape_augs, input_augs
