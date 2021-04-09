"""
DSF-CNN for nuclear segmentation
"""

import tensorflow as tf

from tensorpack import *
from tensorpack.models import BNReLU, Conv2D, MaxPooling
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary

from model.utils.model_utils import *
from model.utils.gconv_utils import *

import sys

sys.path.append("..")  # adds higher directory to python modules path.
try:  # HACK: import beyond current level, may need to restructure
    from config import Config
except ImportError:
    assert False, "Fail to import config.py"


def upsample(name, x, size):
    return tf.image.resize_images(x, [size, size])


def group_concat(x, y, nr_orients):
    shape1 = x.get_shape().as_list()
    chans1 = shape1[3]
    c1 = int(chans1 / nr_orients)
    x = tf.reshape(x, [-1, shape1[1], shape1[2], nr_orients, c1])

    shape2 = y.get_shape().as_list()
    chans2 = shape2[3]
    c2 = int(chans2 / nr_orients)
    y = tf.reshape(y, [-1, shape2[1], shape2[2], nr_orients, c2])

    z = tf.concat([x, y], axis=-1)

    return tf.reshape(z, [-1, shape1[1], shape1[2], nr_orients * (c1 + c2)])


def g_dense_blk(
    name,
    l,
    ch,
    ksize,
    count,
    nr_orients,
    filter_type,
    basis_filter_list,
    rot_matrix_list,
    padding="same",
    bn_init=True,
):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope("blk/" + str(i)):
                if bn_init:
                    x = GBNReLU("preact_bna", l, nr_orients)
                else:
                    x = l
                x = GConv2D(
                    "conv1",
                    x,
                    ch[0],
                    ksize[0],
                    nr_orients,
                    filter_type,
                    basis_filter_list[0],
                    rot_matrix_list[0],
                )
                x = GConv2D(
                    "conv2",
                    x,
                    ch[1],
                    ksize[1],
                    nr_orients,
                    filter_type,
                    basis_filter_list[1],
                    rot_matrix_list[1],
                    activation="identity",
                )
                ##
                if padding == "valid":
                    x_shape = x.get_shape().as_list()
                    l_shape = l.get_shape().as_list()
                    l = crop_op(
                        l,
                        (l_shape[1] - x_shape[2], l_shape[1] - x_shape[2]),
                        "channels_last",
                    )

                l = group_concat(l, x, nr_orients)
        l = GBNReLU("blk_bna", l, nr_orients)
    return l


def encoder(
    name, i, basis_filter_list, rot_matrix_list, nr_orients, filter_type, is_training
):
    """
    Dense Steerable Filter Encoder
    """

    dense_basis_list = [basis_filter_list[1], basis_filter_list[0]]
    dense_rot_list = [rot_matrix_list[1], rot_matrix_list[0]]

    with tf.variable_scope(name):

        c1 = GConv2D(
            "ds_conv1",
            i,
            10,
            7,
            nr_orients,
            filter_type,
            basis_filter_list[1],
            rot_matrix_list[1],
            input_layer=True,
        )
        c2 = GConv2D(
            "ds_conv2",
            c1,
            10,
            7,
            nr_orients,
            filter_type,
            basis_filter_list[1],
            rot_matrix_list[1],
            activation="identity",
        )
        p1 = MaxPooling("max_pool1", c2, 2)
        ####

        d1 = g_dense_blk(
            "dense1",
            p1,
            [14, 6],
            [7, 5],
            3,
            nr_orients,
            filter_type,
            dense_basis_list,
            dense_rot_list,
        )
        c3 = GConv2D(
            "ds_conv3",
            d1,
            16,
            5,
            nr_orients,
            filter_type,
            basis_filter_list[0],
            rot_matrix_list[0],
            activation="identity",
        )
        p2 = MaxPooling("max_pool2", c3, 2, padding="valid")
        ####

        d2 = g_dense_blk(
            "dense2",
            p2,
            [14, 6],
            [7, 5],
            4,
            nr_orients,
            filter_type,
            dense_basis_list,
            dense_rot_list,
        )
        c4 = GConv2D(
            "ds_conv4",
            d2,
            32,
            5,
            nr_orients,
            filter_type,
            basis_filter_list[0],
            rot_matrix_list[0],
            activation="identity",
        )
        p3 = MaxPooling("max_pool3", c4, 2, padding="valid")
        ####

        d3 = g_dense_blk(
            "dense3",
            p3,
            [14, 6],
            [7, 5],
            5,
            nr_orients,
            filter_type,
            dense_basis_list,
            dense_rot_list,
        )
        c5 = GConv2D(
            "ds_conv5",
            d3,
            32,
            5,
            nr_orients,
            filter_type,
            basis_filter_list[0],
            rot_matrix_list[0],
            activation="identity",
        )
        p4 = MaxPooling("max_pool4", c5, 2, padding="valid")
        ####

        d4 = g_dense_blk(
            "dense4",
            p4,
            [14, 6],
            [7, 5],
            6,
            nr_orients,
            filter_type,
            dense_basis_list,
            dense_rot_list,
        )
        c6 = GConv2D(
            "ds_conv6",
            d4,
            32,
            5,
            nr_orients,
            filter_type,
            basis_filter_list[0],
            rot_matrix_list[0],
            activation="identity",
        )

        return [c2, c3, c4, c5, c6]


def decoder(
    name, i, basis_filter_list, rot_matrix_list, nr_orients, filter_type, is_training
):
    """
    Dense Steerable Filter Decoder
    """

    dense_basis_list = [basis_filter_list[1], basis_filter_list[0]]
    dense_rot_list = [rot_matrix_list[1], rot_matrix_list[0]]

    with tf.variable_scope(name):
        with tf.variable_scope("us1"):
            us1 = upsample("us1", i[-1], 32)
            us1_sum = tf.add_n([us1, i[-2]])
            us1 = g_dense_blk(
                "dense_us1",
                us1_sum,
                [14, 6],
                [7, 5],
                4,
                nr_orients,
                filter_type,
                dense_basis_list,
                dense_rot_list,
            )
            us1 = GConv2D(
                "us_conv1",
                us1,
                32,
                5,
                nr_orients,
                filter_type,
                basis_filter_list[0],
                rot_matrix_list[0],
                activation="identity",
            )
        ####

        with tf.variable_scope("us2"):
            us2 = upsample("us2", us1, 64)
            us2_sum = tf.add_n([us2, i[-3]])
            us2 = g_dense_blk(
                "dense_us2",
                us2_sum,
                [14, 6],
                [7, 5],
                3,
                nr_orients,
                filter_type,
                dense_basis_list,
                dense_rot_list,
            )
            us2 = GConv2D(
                "us_conv2",
                us2,
                16,
                5,
                nr_orients,
                filter_type,
                basis_filter_list[0],
                rot_matrix_list[0],
                activation="identity",
            )
        ####

        with tf.variable_scope("us3"):
            us3 = upsample("us3", us2, 128)
            us3_sum = tf.add_n([us3, i[-4]])
            us3 = g_dense_blk(
                "dense_us3",
                us3_sum,
                [14, 6],
                [7, 5],
                2,
                nr_orients,
                filter_type,
                dense_basis_list,
                dense_rot_list,
            )
            us3 = GConv2D(
                "us_conv3",
                us3,
                10,
                5,
                nr_orients,
                filter_type,
                basis_filter_list[0],
                rot_matrix_list[0],
                activation="identity",
            )
        ####

        with tf.variable_scope("us4"):
            us4 = upsample("us4", us3, 256)
            us4_sum = tf.add_n([us4, i[-5]])
            us4 = GConv2D(
                "us_conv4",
                us4_sum,
                10,
                7,
                nr_orients,
                filter_type,
                basis_filter_list[1],
                rot_matrix_list[1],
            )
            feat = GroupPool("us4", us4, nr_orients, pool_type="max")

    return feat


class Model(ModelDesc, Config):
    def __init__(self, freeze=False):
        super(Model, self).__init__()
        # assert tf.test.is_gpu_available()
        self.freeze = freeze
        self.data_format = "NHWC"

    def _get_inputs(self):
        return [
            InputDesc(tf.float32, [None] + self.train_input_shape + [3], "images"),
            InputDesc(
                tf.float32, [None] + self.train_output_shape + [None], "truemap-coded"
            ),
        ]

    # for node to receive manual info such as learning rate.
    def add_manual_variable(self, name, init_value, summary=True):
        var = tf.get_variable(name, initializer=init_value, trainable=False)
        if summary:
            tf.summary.scalar(name + "-summary", var)
        return

    def _get_optimizer(self):
        with tf.variable_scope("", reuse=True):
            lr = tf.get_variable("learning_rate")
        opt = self.optimizer(learning_rate=lr)
        return opt


class Graph(Model):
    def _build_graph(self, inputs):

        is_training = get_current_tower_context().is_training

        images, truemap_coded = inputs
        orig_imgs = images

        pen_map = truemap_coded[..., -1]

        true_np = truemap_coded[..., 0]
        true_np = tf.cast(true_np, tf.int32)
        true_np = tf.identity(true_np, name="truemap-np")
        one_np = tf.one_hot(true_np, 2, axis=-1)
        true_np = tf.expand_dims(true_np, axis=-1)

        true_mk = truemap_coded[..., 1:4]
        true_mk = tf.cast(true_mk, tf.int32)
        true_mk = tf.identity(true_mk, name="truemap-mk")
        one_mk = tf.cast(true_mk, tf.float32)

        ####
        with argscope(
            Conv2D,
            activation=tf.identity,
            use_bias=False,  # K.he initializer
            W_init=tf.variance_scaling_initializer(scale=2.0, mode="fan_out"),
        ), argscope([Conv2D], data_format=self.data_format):

            i = images if not self.input_norm else images / 255.0

            ####
            d = encoder(
                "encoder",
                i,
                self.basis_filter_list,
                self.rot_matrix_list,
                self.nr_orients,
                self.filter_type,
                is_training,
            )
            ####
            feat = decoder(
                "decoder",
                d,
                self.basis_filter_list,
                self.rot_matrix_list,
                self.nr_orients,
                self.filter_type,
                is_training,
            )

            feat_np = Conv2D("feat_np", feat, 96, 1, use_bias=True, nl=BNReLU)
            o_logi_np = Conv2D(
                "output_np", feat_np, 2, 1, use_bias=True, nl=tf.identity
            )
            soft_np = tf.nn.softmax(o_logi_np, axis=-1)
            prob_np = tf.identity(soft_np[..., 1], name="predmap-prob")
            prob_np = tf.expand_dims(prob_np, -1)

            feat_mk = Conv2D("feat_mk", feat, 96, 1, use_bias=True, nl=BNReLU)
            o_logi_mk = Conv2D(
                "output_mk", feat_mk, 3, 1, use_bias=True, nl=tf.identity
            )
            soft_mk = tf.nn.softmax(o_logi_mk, axis=-1)
            prob_mk = tf.identity(soft_mk[..., :2], name="predmap-prob")

            # encoded so that inference can extract all output at once
            predmap_coded = tf.concat([prob_np, prob_mk], axis=-1, name="predmap-coded")

        ####
        if get_current_tower_context().is_training:
            # ---- LOSS ----#
            loss = 0
            for term, weight in self.loss_term.items():
                if term == "bce":
                    term_loss_np = categorical_crossentropy(soft_np, one_np)
                    term_loss_np = tf.reduce_mean(term_loss_np, name="loss-bce-np")

                    term_loss_mk = categorical_crossentropy(soft_mk, one_mk)
                    term_loss_mk = tf.reduce_mean(
                        term_loss_mk * pen_map, name="loss-bce-mk"
                    )
                elif "dice" in self.loss_term:
                    # branch 1
                    term_loss_np = dice_loss(
                        soft_np[..., 0], one_np[..., 0]
                    ) + dice_loss(soft_np[..., 1], one_np[..., 1])
                    term_loss_np = tf.identity(term_loss_np, name="loss-dice-np")

                    term_loss_mk = dice_loss(
                        soft_mk[..., 0], one_mk[..., 0]
                    ) + dice_loss(soft_mk[..., 1], one_mk[..., 1])
                    term_loss_mk = tf.identity(term_loss_mk, name="loss-dice-mk")
                else:
                    assert False, "Not support loss term: %s" % term
                add_moving_summary(term_loss_np)
                add_moving_summary(term_loss_mk)
                loss += term_loss_np + term_loss_mk

            ### combine the loss into single cost function
            wd_loss = regularize_cost(".*/W", l2_regularizer(1.0e-7), name="l2_wd_loss")
            add_moving_summary(wd_loss)
            self.cost = tf.identity(loss + wd_loss, name="overall-loss")
            add_moving_summary(self.cost)
            ####

            add_param_summary((".*/W", ["histogram"]))  # monitor W

            ### logging visual sthg
            orig_imgs = tf.cast(orig_imgs, tf.uint8)
            tf.summary.image("input", orig_imgs, max_outputs=1)

            pred_np = colorize(prob_np[..., 0], cmap="jet")
            true_np = colorize(true_np[..., 0], cmap="jet")

            pred_mk_blb = colorize(prob_mk[..., 0], cmap="jet")
            true_mk_blb = colorize(true_mk[..., 0], cmap="jet")
            pred_mk_cnt = colorize(prob_mk[..., 1], cmap="jet")
            true_mk_cnt = colorize(true_mk[..., 1], cmap="jet")

            viz = tf.concat(
                [
                    orig_imgs,
                    pred_np,
                    pred_mk_blb,
                    pred_mk_cnt,
                    true_np,
                    true_mk_blb,
                    true_mk_cnt,
                ],
                2,
            )

            viz = tf.concat([viz[0], viz[-1]], axis=0)
            viz = tf.expand_dims(viz, axis=0)
            tf.summary.image("output", viz, max_outputs=1)

        return
