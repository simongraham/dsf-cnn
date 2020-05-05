import tensorflow as tf

from tensorpack import *
from tensorpack.models import BatchNorm, BNReLU, Conv2D, MaxPooling, FixedUnPooling
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary

from model.utils.model_utils import *
from model.utils.gconv_utils import *

import sys
sys.path.append("..") # adds higher directory to python modules path.
try: # HACK: import beyond current level, may need to restructure
    from config import Config
except ImportError:
    assert False, 'Fail to import config.py'


def group_concat(x, y, nr_orients):
    shape1 = x.get_shape().as_list()
    chans1 = shape1[3]
    c1 = int(chans1/nr_orients)
    x = tf.reshape(x, [-1,shape1[1],shape1[2], nr_orients, c1])

    shape2 = y.get_shape().as_list()
    chans2 = shape2[3]
    c2 = int(chans2/nr_orients)
    y = tf.reshape(y, [-1,shape2[1],shape2[2], nr_orients, c2])

    z = tf.concat([x, y], axis=-1)

    return tf.reshape(z, [-1,shape1[1],shape1[2],nr_orients*(c1+c2)])

####
def g_dense_blk(name, l, ch, ksize, count, nr_orients, filter_type, basis_filter_list, rot_matrix_list, padding='same', bn_init=True):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('blk/' + str(i)):
                if bn_init:
                    x = GBNReLU('preact_bna', l, nr_orients)
                else:
                    x = l
                x = GConv2D('conv1', x, ch[0], ksize[0], nr_orients, filter_type, basis_filter_list[0], rot_matrix_list[0])
                x = GConv2D('conv2', x, ch[1], ksize[1], nr_orients, filter_type, basis_filter_list[1], rot_matrix_list[1], activation=False)
                ##
                if padding == 'valid':
                    x_shape = x.get_shape().as_list()
                    l_shape = l.get_shape().as_list()
                    l = crop_op(l, (l_shape[1] - x_shape[2], 
                                    l_shape[1] - x_shape[2]), 'channels_last')

                l = group_concat(l, x, nr_orients)
        l = GBNReLU('blk_bna', l, nr_orients)
    return l
####

def net(name, i, basis_filter_list, rot_matrix_list, nr_orients, filter_type, is_training):
    '''
    Steerable group equivariant encoder
    '''
    dense_basis_list = [basis_filter_list[0],basis_filter_list[1]]
    dense_rot_list = [rot_matrix_list[0], rot_matrix_list[1]]

    with tf.variable_scope(name):

        c1 = GConv2D('ds_conv1', i, 8, 7, nr_orients, filter_type, basis_filter_list[1], rot_matrix_list[1], input_layer=True)
        c2 = GConv2D('ds_conv2', c1, 8, 7, nr_orients, filter_type, basis_filter_list[1], rot_matrix_list[1])
        p1 = MaxPooling('max_pool1', c2, 2)  
        ####
        
        d1 = g_dense_blk('dense1', p1, [32,8], [5,7], 2, nr_orients, filter_type, dense_basis_list, dense_rot_list, bn_init=False)
        c3 = GConv2D('ds_conv3', d1, 32, 5, nr_orients, filter_type, basis_filter_list[0], rot_matrix_list[0])
        p2 = MaxPooling('max_pool2', c3, 2, padding= 'valid') 
        ####

        d2 = g_dense_blk('dense2', p2, [32,8], [5,7], 2, nr_orients, filter_type, dense_basis_list, dense_rot_list, bn_init=False)
        c4 = GConv2D('ds_conv4', d2, 32, 5, nr_orients, filter_type, basis_filter_list[0], rot_matrix_list[0])
        p3 = MaxPooling('max_pool3', c4, 2, padding= 'valid') 
        ####

        d3 = g_dense_blk('dense3', p3, [32,8], [5,7], 3, nr_orients, filter_type, dense_basis_list, dense_rot_list, bn_init=False)
        c5 = GConv2D('ds_conv5', d3, 32, 5, nr_orients, filter_type, basis_filter_list[0], rot_matrix_list[0])
        p4 = MaxPooling('max_pool4', c5, 2, padding= 'valid')  
        ####

        d4 = g_dense_blk('dense4', p4, [32,8], [5,7], 3, nr_orients, filter_type, dense_basis_list, dense_rot_list, bn_init=False)
        c6 = GConv2D('ds_conv6', d4, 32, 5, nr_orients, filter_type, basis_filter_list[0], rot_matrix_list[0])
        p5 = AvgPooling('glb_avg_pool', c6, 6, padding= 'valid')
        p6 = GroupPool('orient_pool', p5, nr_orients, pool_type='max')
        ####

        c7 = Conv2D('conv3', p6, 96, 1, use_bias=True, nl=BNReLU)
        c7 = tf.layers.dropout(c7, rate=0.3, seed=5, training=is_training)
        c8 = Conv2D('conv4', c7, 96, 1, use_bias=True, nl=BNReLU)
        c8 = tf.layers.dropout(c8, rate=0.3, seed=5, training=is_training)

        return c8

####
class Model(ModelDesc, Config):
    def __init__(self, freeze=False):
        super(Model, self).__init__()
        # assert tf.test.is_gpu_available()
        self.freeze = freeze
        self.data_format = 'NHWC'

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None] + self.train_input_shape + [3], 'images'),
                InputDesc(tf.float32, [None] + self.train_output_shape + [None], 'truemap-coded')]
    
    # for node to receive manual info such as learning rate.
    def add_manual_variable(self, name, init_value, summary=True):
        var = tf.get_variable(name, initializer=init_value, trainable=False)
        if summary:
            tf.summary.scalar(name + '-summary', var)
        return

    def _get_optimizer(self):
        with tf.variable_scope("", reuse=True):
            lr = tf.get_variable('learning_rate')
        opt = self.optimizer(learning_rate=lr)
        return opt

####
class Graph(Model):
    def _build_graph(self, inputs):

        is_training = get_current_tower_context().is_training
        
        images, truemap_coded = inputs
        orig_imgs = images
        true = truemap_coded[...,0]
        true = tf.cast(true, tf.int32)
        true = tf.identity(true, name='truemap')
        one_hot  = tf.one_hot(true, 2, axis=-1)
        true = tf.expand_dims(true, axis=-1)

        ####
        with argscope(Conv2D, activation=tf.identity, use_bias=False, # K.he initializer
                      W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')), \
                argscope([Conv2D, BatchNorm], data_format=self.data_format):

            i = images if not self.input_norm else images / 255.0

            ####
            feat = net('net', i, self.basis_filter_list, self.rot_matrix_list, self.nr_orients, self.filter_type, is_training)

            #### Prediction
            o_logi = Conv2D('output', feat, 2, 1, use_bias=True, nl=tf.identity)
            soft = tf.nn.softmax(o_logi, axis=-1)

            prob = tf.identity(soft, name='predmap-prob')
     
            # encoded so that inference can extract all output at once
            predmap_coded = tf.concat(prob, axis=-1, name='predmap-coded')

        ####
        if get_current_tower_context().is_training:
            #---- LOSS ----#
            loss = 0
            for term, weight in self.loss_term.items():
                if term == 'bce':
                    term_loss = categorical_crossentropy(soft, one_hot)
                    term_loss = tf.reduce_mean(term_loss, name='loss-bce')
                else:
                    assert False, 'Not support loss term: %s' % term
                add_moving_summary(term_loss)
                loss += term_loss * weight

            ### combine the loss into single cost function
            wd_loss = regularize_cost('.*/W', l2_regularizer(1.0e-7), name='l2_wd_loss')
            add_moving_summary(wd_loss)
            self.cost = tf.identity(loss+wd_loss, name='overall-loss')            
            add_moving_summary(self.cost)
            ####

            add_param_summary(('.*/W', ['histogram']))   # monitor W

            ### logging visual sthg
            orig_imgs = tf.cast(orig_imgs  , tf.uint8)
            tf.summary.image('input', orig_imgs, max_outputs=1)

        return
####