# -*- coding: utf-8 -*-
# File: custom_ops.py
###
# https://github.com/tensorpack/tensorpack/blob/master/tensorpack/models/batch_norm.py
###

import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages
import re
import six
import functools

from tensorpack.utils import logger
from tensorpack.utils.argtools import get_data_format
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.tfutils.collection import backup_collection, restore_collection
from tensorpack import layer_register, VariableHolder
from tensorpack.tfutils.varreplace import custom_getter_scope

def rename_get_variable(mapping):
    """
    Args:
        mapping(dict): an old -> new mapping for variable basename. e.g. {'kernel': 'W'}
    Returns:
        A context where the variables are renamed.
    """
    def custom_getter(getter, name, *args, **kwargs):
        splits = name.split('/')
        basename = splits[-1]
        if basename in mapping:
            basename = mapping[basename]
            splits[-1] = basename
            name = '/'.join(splits)
        return getter(name, *args, **kwargs)
    return custom_getter_scope(custom_getter)

def map_common_tfargs(kwargs):
    df = kwargs.pop('data_format', None)
    if df is not None:
        df = get_data_format(df, tfmode=True)
        kwargs['data_format'] = df

    old_nl = kwargs.pop('nl', None)
    if old_nl is not None:
        kwargs['activation'] = lambda x, name=None: old_nl(x, name=name)

    if 'W_init' in kwargs:
        kwargs['kernel_initializer'] = kwargs.pop('W_init')

    if 'b_init' in kwargs:
        kwargs['bias_initializer'] = kwargs.pop('b_init')
    return kwargs

def convert_to_tflayer_args(args_names, name_mapping):
    """
    After applying this decorator:
    1. data_format becomes tf.layers style
    2. nl becomes activation
    3. initializers are renamed
    4. positional args are transformed to correspoding kwargs, according to args_names
    5. kwargs are mapped to tf.layers names if needed, by name_mapping
    """

    def decorator(func):
        @functools.wraps(func)
        def decorated_func(inputs, *args, **kwargs):
            kwargs = map_common_tfargs(kwargs)

            posarg_dic = {}
            assert len(args) <= len(args_names), \
                "Please use kwargs instead of positional args to call this model, " \
                "except for the following arguments: {}".format(', '.join(args_names))
            for pos_arg, name in zip(args, args_names):
                posarg_dic[name] = pos_arg

            ret = {}
            for name, arg in six.iteritems(kwargs):
                newname = name_mapping.get(name, None)
                if newname is not None:
                    assert newname not in kwargs, \
                        "Argument {} and {} conflicts!".format(name, newname)
                else:
                    newname = name
                ret[newname] = arg
            ret.update(posarg_dic)  # Let pos arg overwrite kw arg, for argscope to work

            return func(inputs, **ret)

        return decorated_func

    return decorator

__all__ = ['BatchNorm3d', 'BatchNormVec']


def get_bn_variables(n_out, use_scale, use_bias, beta_init, gamma_init):
    if use_bias:
        beta = tf.get_variable('beta', [n_out], initializer=beta_init)
    else:
        beta = tf.zeros([n_out], name='beta')
    if use_scale:
        gamma = tf.get_variable('gamma', [n_out], initializer=gamma_init)
    else:
        gamma = tf.ones([n_out], name='gamma')
    # x * gamma + beta

    moving_mean = tf.get_variable('mean/EMA', [n_out],
                                  initializer=tf.constant_initializer(), trainable=False)
    moving_var = tf.get_variable('variance/EMA', [n_out],
                                 initializer=tf.constant_initializer(1.0), trainable=False)

    if get_current_tower_context().is_main_training_tower:
        for v in [moving_mean, moving_var]:
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)
    return beta, gamma, moving_mean, moving_var


def internal_update_bn_ema(xn, batch_mean, batch_var,
                           moving_mean, moving_var, decay):
    update_op1 = moving_averages.assign_moving_average(
        moving_mean, batch_mean, decay, zero_debias=False,
        name='mean_ema_op')
    update_op2 = moving_averages.assign_moving_average(
        moving_var, batch_var, decay, zero_debias=False,
        name='var_ema_op')

    # When sync_statistics is True, always enable internal_update.
    # Otherwise the update ops (only executed on main tower)
    # will hang when some BatchNorm layers are unused (https://github.com/tensorpack/tensorpack/issues/1078)
    with tf.control_dependencies([update_op1, update_op2]):
        return tf.identity(xn, name='output')

@layer_register()
@convert_to_tflayer_args(
    args_names=[],
    name_mapping={
        'use_bias': 'center',
        'use_scale': 'scale',
        'gamma_init': 'gamma_initializer',
        'decay': 'momentum',
        'use_local_stat': 'training'
    })
def BatchNorm3d(inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,
              center=True, scale=True,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              virtual_batch_size=None,
              data_format='channels_last',
              internal_update=False,
              sync_statistics=None):
    """
    Almost equivalent to `tf.layers.batch_normalization`, but different (and more powerful)
    in the following:
    1. Accepts an alternative `data_format` option when `axis` is None. For 2D input, this argument will be ignored.
    2. Default value for `momentum` and `epsilon` is different.
    3. Default value for `training` is automatically obtained from tensorpack's `TowerContext`, but can be overwritten.
    4. Support the `internal_update` option, which enables the use of BatchNorm layer inside conditionals.
    5. Support the `sync_statistics` option, which is very useful in small-batch models.
    Args:
        internal_update (bool): if False, add EMA update ops to
          `tf.GraphKeys.UPDATE_OPS`. If True, update EMA inside the layer by control dependencies.
          They are very similar in speed, but `internal_update=True` can be used
          when you have conditionals in your model, or when you have multiple networks to train.
          Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/14699
        sync_statistics: either None or "nccl". By default (None), it uses statistics of the input tensor to normalize.
          When set to "nccl", this layer must be used under tensorpack multi-gpu trainers,
          and it then uses per-machine (multiple GPU) statistics to normalize.
          Note that this implementation averages the per-tower E[x] and E[x^2] among towers to compute
          global mean&variance. The result is the global mean&variance only if each tower has the same batch size.
          This option has no effect when not training.
          This option is also known as "Cross-GPU BatchNorm" as mentioned in https://arxiv.org/abs/1711.07240.
          Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/18222
    Variable Names:
    * ``beta``: the bias term. Will be zero-inited by default.
    * ``gamma``: the scale term. Will be one-inited by default.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.
    Note:
        Combinations of ``training`` and ``ctx.is_training``:
        * ``training == ctx.is_training``: standard BN, EMA are maintained during training
          and used during inference. This is the default.
        * ``training and not ctx.is_training``: still use batch statistics in inference.
        * ``not training and ctx.is_training``: use EMA to normalize in
          training. This is useful when you load a pre-trained BN and
          don't want to fine tune the EMA. EMA will not be updated in
          this case.
    """
    # parse shapes
    data_format = get_data_format(data_format, tfmode=False)
    shape = inputs.get_shape().as_list()
    ndims = len(shape)
    # in 3d conv, we have 5d dim [batch, h, w, t, c]
    if sync_statistics is not None:
        sync_statistics = sync_statistics.lower()
    assert sync_statistics in [None, 'nccl', 'horovod'], sync_statistics

    if axis is None:
        assert ndims == 5, 'Number of input dims must be 5 for 3d Batch Norm'
        axis = 1 if data_format == 'NCHW' else 4

    num_chan = shape[axis]

    # parse training/ctx
    ctx = get_current_tower_context()
    if training is None:
        training = ctx.is_training
    training = bool(training)
    TF_version = get_tf_version_tuple()
    if not training and ctx.is_training:
        assert TF_version >= (1, 4), \
            "Fine tuning a BatchNorm model with fixed statistics is only " \
            "supported after https://github.com/tensorflow/tensorflow/pull/12580 "
        if ctx.is_main_training_tower:  # only warn in first tower
            logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
        # Using moving_mean/moving_variance in training, which means we
        # loaded a pre-trained BN and only fine-tuning the affine part.

    if sync_statistics is None or not (training and ctx.is_training):
        coll_bk = backup_collection([tf.GraphKeys.UPDATE_OPS])
        with rename_get_variable(
                {'moving_mean': 'mean/EMA',
                 'moving_variance': 'variance/EMA'}):
            tf_args = dict(
                axis=axis,
                momentum=momentum, epsilon=epsilon,
                center=center, scale=scale,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                fused=True,
                _reuse=tf.get_variable_scope().reuse)
            if TF_version >= (1, 5):
                tf_args['virtual_batch_size'] = virtual_batch_size
            else:
                assert virtual_batch_size is None, "Feature not supported in this version of TF!"
            layer = tf.layers.BatchNormalization(**tf_args)
            xn = layer.apply(inputs, training=training, scope=tf.get_variable_scope())

        # maintain EMA only on one GPU is OK, even in replicated mode.
        # because during training, EMA isn't used
        if ctx.is_main_training_tower:
            for v in layer.non_trainable_variables:
                add_model_variable(v)
        if not ctx.is_main_training_tower or internal_update:
            restore_collection(coll_bk)

        if training and internal_update:
            assert layer.updates
            with tf.control_dependencies(layer.updates):
                ret = tf.identity(xn, name='output')
        else:
            ret = tf.identity(xn, name='output')

        vh = ret.variables = VariableHolder(
            moving_mean=layer.moving_mean,
            mean=layer.moving_mean,  # for backward-compatibility
            moving_variance=layer.moving_variance,
            variance=layer.moving_variance)  # for backward-compatibility
        if scale:
            vh.gamma = layer.gamma
        if center:
            vh.beta = layer.beta
    else:
        red_axis = [0] if ndims == 2 else ([0, 2, 3] if axis == 1 else [0, 1, 2])
        if ndims == 5:
            red_axis = [0, 2, 3, 4] if axis == 1 else [0, 1, 2, 3]
        new_shape = None  # don't need to reshape unless ...
        if ndims == 4 and axis == 1:
            new_shape = [1, num_chan, 1, 1]
        if ndims == 5 and axis == 1:
            new_shape = [1, num_chan, 1, 1, 1]

        batch_mean = tf.reduce_mean(inputs, axis=red_axis)
        batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axis)

        if sync_statistics == 'nccl':
            if six.PY3 and TF_version <= (1, 8) and ctx.is_main_training_tower:
                logger.warn("A TensorFlow bug will cause cross-GPU BatchNorm to fail. "
                            "Apply this patch: https://github.com/tensorflow/tensorflow/pull/20360")

            from tensorflow.contrib.nccl.ops import gen_nccl_ops
            shared_name = re.sub('tower[0-9]+/', '', tf.get_variable_scope().name)
            num_dev = ctx.total
            batch_mean = gen_nccl_ops.nccl_all_reduce(
                input=batch_mean,
                reduction='sum',
                num_devices=num_dev,
                shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
            batch_mean_square = gen_nccl_ops.nccl_all_reduce(
                input=batch_mean_square,
                reduction='sum',
                num_devices=num_dev,
                shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
        elif sync_statistics == 'horovod':
            # Require https://github.com/uber/horovod/pull/331
            # Proof-of-concept, not ready yet.
            import horovod.tensorflow as hvd
            batch_mean = hvd.allreduce(batch_mean, average=True)
            batch_mean_square = hvd.allreduce(batch_mean_square, average=True)
        batch_var = batch_mean_square - tf.square(batch_mean)
        batch_mean_vec = batch_mean
        batch_var_vec = batch_var

        beta, gamma, moving_mean, moving_var = get_bn_variables(
            num_chan, scale, center, beta_initializer, gamma_initializer)
        if new_shape is not None:
            batch_mean = tf.reshape(batch_mean, new_shape)
            batch_var = tf.reshape(batch_var, new_shape)
            # Using fused_batch_norm(is_training=False) is actually slightly faster,
            # but hopefully this call will be JITed in the future.
            xn = tf.nn.batch_normalization(
                inputs, batch_mean, batch_var,
                tf.reshape(beta, new_shape),
                tf.reshape(gamma, new_shape), epsilon)
        else:
            xn = tf.nn.batch_normalization(
                inputs, batch_mean, batch_var,
                beta, gamma, epsilon)

        if ctx.is_main_training_tower:
            ret = update_bn_ema(
                xn, batch_mean_vec, batch_var_vec, moving_mean, moving_var,
                momentum, internal_update)
        else:
            ret = tf.identity(xn, name='output')

        vh = ret.variables = VariableHolder(
            moving_mean=moving_mean,
            mean=moving_mean,  # for backward-compatibility
            moving_variance=moving_var,
            variance=moving_var)  # for backward-compatibility
        if scale:
            vh.gamma = gamma
        if center:
            vh.beta = beta
    return ret

    
@layer_register()
@convert_to_tflayer_args(
    args_names=[],
    name_mapping={
        'use_bias': 'center',
        'use_scale': 'scale',
        'gamma_init': 'gamma_initializer',
        'decay': 'momentum',
        'use_local_stat': 'training'
    })

def BatchNormVec(inputs, axis=None, training=None, momentum=0.1, epsilon=1e-5,
              center=False, scale=True,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              virtual_batch_size=None,
              data_format='channels_last',
              ema_update='default',
              sync_statistics=None,
              internal_update=None):
    """
    A more powerful version of `tf.layers.batch_normalization`. It differs from
    the offical one in the following aspects:
    """
    # parse training/ctx
    ctx = get_current_tower_context()
    if training is None:
        training = ctx.is_training
    training = bool(training)

    # parse shapes
    data_format = get_data_format(data_format)
    shape = inputs.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4], ndims
    if sync_statistics is not None:
        sync_statistics = sync_statistics.lower()
    assert sync_statistics in [None, 'nccl', 'horovod'], sync_statistics

    assert ema_update in ["default", "collection", "internal", "skip"]
    if internal_update is not None:
        log_deprecated("BatchNorm(internal_update=)", "Use ema_update='internal' instead!", "2020-01-01")
        assert ema_update == 'default', \
            "Do not use internal_update and ema_update together! internal_update is deprecated"
        ema_update = "internal" if internal_update else "collection"
    if ema_update == "default":
        ema_update = "collection"
    # Logic:
    # 1. EMA update is possible only when we compute batch statistics (training=True)
    # 2. We know that in training, non-main training tower does not need EMA
    #    update (unless you need, e.g., inference during training on all towers)
    #    We don't know about what to do in prediction context, so be conservative and do the update.
    # 3. User can explicit disable update by "skip".
    do_ema_update = training and \
        (ctx.is_main_training_tower or not ctx.is_training) \
        and (ema_update != "skip")

    if axis is None:
        if ndims == 2:
            axis = 1
        else:
            axis = 1 if data_format == 'NCHW' else 3
    assert axis in [1, 3], axis
    num_chan = shape[axis]

    TF_version = get_tf_version_tuple()

    freeze_bn_backward = not training and ctx.is_training
    if freeze_bn_backward:
        assert TF_version >= (1, 4), \
            "Fine tuning a BatchNorm model with fixed statistics needs TF>=1.4!"
        if ctx.is_main_training_tower:  # only warn in first tower
            log_once("Some BatchNorm layer uses moving_mean/moving_variance in training.", func='warn')
        # Using moving_mean/moving_variance in training, which means we
        # loaded a pre-trained BN and only fine-tuning the affine part.

    do_sync_bn = (sync_statistics is not None) and training

    red_axis = [0] if ndims == 2 else ([0, 2, 3] if axis == 1 else [0, 1, 2])

    new_shape = None  # don't need to reshape unless ...
    if ndims == 4 and axis == 1:
        new_shape = [1, num_chan, 1, 1]
    batch_mean = tf.reduce_mean(inputs, axis=red_axis)
    batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axis)

    batch_var = batch_mean_square - tf.square(batch_mean)
    batch_mean_vec = batch_mean
    batch_var_vec = batch_var
    beta, gamma, moving_mean, moving_var = get_bn_variables(
        num_chan, scale, center, beta_initializer, gamma_initializer)
    if new_shape is not None:
        batch_mean = tf.reshape(batch_mean, new_shape)
        batch_var = tf.reshape(batch_var, new_shape)
        # Using fused_batch_norm(is_training=False) is actually slightly faster,
        # but hopefully this call will be JITed in the future.
        if training:
            xn = tf.nn.batch_normalization(
                inputs, tf.zeros([num_chan]), batch_var,
                tf.reshape(beta, new_shape),
                tf.reshape(gamma, new_shape), epsilon)
        else:
            xn = tf.nn.batch_normalization(
                inputs, tf.zeros([num_chan]), moving_var,
                tf.reshape(beta, new_shape),
                tf.reshape(gamma, new_shape), epsilon)
    else:
        if training:
            xn = tf.nn.batch_normalization(
                inputs, tf.zeros([num_chan]), batch_var,
                beta, gamma, epsilon)
        else:
            xn = tf.nn.batch_normalization(
                inputs, tf.zeros([num_chan]), moving_var,
                beta, gamma, epsilon)

    if do_ema_update:
        ret = internal_update_bn_ema(
            xn, tf.zeros([num_chan]), batch_var_vec, moving_mean, moving_var, momentum)
    else:
        ret = tf.identity(xn, name='output')

    vh = ret.variables = VariableHolder(
        moving_mean=moving_mean,
        mean=moving_mean,  # for backward-compatibility
        moving_variance=moving_var,
        variance=moving_var)  # for backward-compatibility
    if scale:
        vh.gamma = gamma
    if center:
        vh.beta = beta
    return ret
