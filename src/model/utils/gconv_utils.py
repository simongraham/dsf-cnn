import math
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.common import get_tf_version_tuple

from matplotlib import cm

from model.utils.norm_utils import *
from model.utils.rotation_utils import *


####
def GBNReLU(name, x, nr_orients):
    """
    A shorthand of Group Equivariant BatchNormalization + ReLU.

    Args:
        name: variable scope name
        x: input tensor
        nr_orients: number of filter orientations

    Returns:
        out: normalised tensor with ReLU activation
    """

    shape = x.get_shape().as_list()
    chans = shape[3]

    c = int(chans/nr_orients)

    x = tf.reshape(x, [-1, shape[1], shape[2], nr_orients, c])
    bn = BatchNorm3d(name + '_bn', x)
    act = tf.nn.relu(bn, name='relu')
    out = tf.reshape(act, [-1, shape[1], shape[2], chans])
    return out
####


def GBatchNorm(name, x, nr_orients):
    """
    Group Equivariant BatchNormalization.
    
    Args:
        name: variable scope name
        x: input tensor
        nr_orients: number of filter orientations

    Returns:
        out: normalised tensor
    """

    shape = x.get_shape().as_list()
    chans = shape[3]

    c = int(chans/nr_orients)

    x = tf.reshape(x, [-1, shape[1], shape[2], nr_orients, c])
    bn = BatchNorm3d(name + '_bn', x)
    out = tf.reshape(act, [-1, shape[1], shape[2], chans])
    return out
####


def get_basis_params(k_size):
    """
    Get the filter parameters for a given kernel size

    Args:
        k_size (int): input kernel size
    
    Returns:
        alpha_list: list of alpha values
        beta_list:  list of beta values 
        bl_list:    used to bandlimit high frequency filters in get_basis_filters()
    """

    if k_size == 5:
        alpha_list = [0, 1, 2]
        beta_list = [0, 1, 2]
        bl_list = [0, 2, 2]
    if k_size == 7:
        alpha_list = [0, 1, 2, 3]
        beta_list = [0, 1, 2, 3]
        bl_list = [0, 2, 3, 2]
    if k_size == 9:
        alpha_list = [0, 1, 2, 3, 4]
        beta_list = [0, 1, 2, 3, 4]
        bl_list = [0, 3, 4, 4, 3]
    if k_size == 11:
        alpha_list = [0, 1, 2, 3, 4]
        beta_list = [1, 2, 3, 4]
        bl_list = [0, 3, 4, 4, 3]

    return alpha_list, beta_list, bl_list
####


def get_basis_filters(alpha_list, beta_list, bl_list, k_size, eps=10**-8):
    """
    Gets the atomic basis filters

    Args:
        alpha_list: 
        beta_list:
        bl_list: 
        k_size (int): kernel size of basis filters
        eps=10**-8: epsilon used to prevent division by 0
    
    Returns:
        filter_list_bl: list of filters, with bandlimiting (bl) to reduce aliasing
        alpha_list_bl:  corresponding list of alpha used in bandlimited filters
        beta_list_bl:   corresponding list of beta used in bandlimited filters
    """

    filter_list = []
    freq_list = []
    for beta in beta_list:
        for alpha in alpha_list:
            if alpha <= bl_list[beta]:
                his = k_size//2  # half image size
                y_index, x_index = np.mgrid[-his:(his+1), -his:(his+1)]
                y_index *= -1
                z_index = x_index + 1j*y_index

                # convert z to natural coordinates and add eps to avoid division by zero
                z = (z_index + eps)
                r = np.abs(z)

                if beta == beta_list[-1]:
                    sigma = 0.4
                else:
                    sigma = 0.6
                rad_prof = np.exp(-(r-beta)**2/(2*(sigma**2)))
                c_image = rad_prof * (z/r)**alpha
                c_image_norm = (math.sqrt(2)*c_image) / np.linalg.norm(c_image)

                # add basis filter to list
                filter_list.append(c_image)
                # add corresponding frequency of filter to list (info needed for phase manipulation)
                freq_list.append(alpha)

    filter_array = np.array(filter_list)

    filter_array = np.reshape(filter_array, [
                              filter_array.shape[0], filter_array.shape[1], filter_array.shape[2], 1, 1, 1])
    return tf.convert_to_tensor(filter_array, dtype=tf.complex64), freq_list
####


def get_rot_info(nr_orients, freq_list):
    """ 
    Generate rotation info for phase manipulation of steerable filters.

    Args:
        nr_orients: number of filter rotations
        freq_list:

    Returns:
        rot_info:
    """

    # Generate rotation matrix for phase manipulation of steerable function
    rot_list = []
    for i in range(len(freq_list)):
        list_tmp = []
        for j in range(nr_orients):
            # Rotation is dependent on the frequency of the basis filter
            angle = (2*np.math.pi / nr_orients) * j
            list_tmp.append(np.exp(-1j*freq_list[i]*angle))
        rot_list.append(list_tmp)
    rot_info = np.array(rot_list)

    # Reshape to enable matrix multiplication
    rot_info = np.reshape(
        rot_info, [rot_info.shape[0], 1, 1, 1, 1, nr_orients])
    rot_info = tf.convert_to_tensor(rot_info, dtype=tf.complex64)
    return rot_info
####


def GroupPool(name, x, nr_orients, pool_type='max'):
    """
    Perform pooling along the orientation axis. 

    Args:
        name: variable scope name
        x: input tensor
        nr_orients: number of filter orientations
        pool_type: choose either 'max' or 'mean'

    Returns:
        pool: pooled tensor
    """
    shape = x.get_shape().as_list()
    new_shape = [-1, shape[1], shape[2], nr_orients, shape[3] // nr_orients]
    x_reshape = tf.reshape(x, new_shape)
    if pool_type == 'max':
        pool = tf.reduce_max(x_reshape, 3)
    elif pool_type == 'mean':
        pool = tf.reduce_mean(x_reshape, 3)
    else:
        raise ValueError('Pool type not recognised')
    return pool
####


def steerable_initializer(input_layer, nr_orients, factor=2.0, mode='FAN_IN', uniform=False,
                          seed=None, dtype=dtypes.float32):
    """
    Initialise complex coefficients in accordance with Weiler et al. (https://arxiv.org/pdf/1711.07289.pdf)
    Note, here we use the truncated normal dist, whereas Weiler et al. uses the regular normal dist.

    Args:
        input_layer:
        nr_orients: number of filter orientations
        factor:
        mode:
        uniform: 
        seed:
        dtype:

    Returns:
        _initializer: 
    """
    def _initializer(shape, dtype=dtype, partition_info=None):

        # total number of basis filters
        Q = shape[0]*shape[1]
        if mode == 'FAN_IN':
            fan_in = shape[-2]
            C = fan_in
            # count number of input connections.
        elif mode == 'FAN_OUT':
            fan_out = shape[-2]
            # count number of output connections.
            C = fan_out
        n = C*Q
        # to get stddev = math.sqrt(factor / n) need to adjust for truncated.
        trunc_stddev = math.sqrt(factor / n) / .87962566103423978
        return random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                           seed=seed)

    return _initializer
####


def cycle_channels(filters, shape_list):
    """
    Perform cyclic permutation of the orientation channels for kernels on the group G.

    Args:
        filters: 
        shape_list:

    Returns:
        tensor of filters with channels permuted
    """
    # Shape list = [nr_orients_out, K, K, nr_orients_in, filters_in, filters_out]
    nr_orients_out = shape_list[0]
    rotated_filters = [None] * nr_orients_out
    for orientation in range(nr_orients_out):
        # [K, K, nr_orients_in, filters_in, filters_out]
        filters_temp = filters[orientation]
        # [K, K, filters_in, filters_out, nr_orients]
        filters_temp = tf.transpose(filters_temp, [0, 1, 3, 4, 2])
        # [K * K * filters_in * filters_out, nr_orients_in]
        filters_temp = tf.reshape(
            filters_temp, [shape_list[1] * shape_list[2] * shape_list[4] * shape_list[5], shape_list[3]])
        # Cycle along the orientation axis
        roll_matrix = tf.constant(
            np.roll(np.identity(shape_list[3]), orientation, axis=1), dtype=tf.float32)
        filters_temp = tf.matmul(filters_temp, roll_matrix)
        filters_temp = tf.reshape(
            filters_temp, [shape_list[1], shape_list[2], shape_list[4], shape_list[5], shape_list[3]])
        filters_temp = tf.transpose(filters_temp, [0, 1, 4, 2, 3])
        rotated_filters[orientation] = filters_temp

    return tf.stack(rotated_filters)
####


def gen_rotated_filters(w, filter_type, input_layer, nr_orients_out, basis_filters=None, rot_info=None):
    """
    Generate the rotated filters either by phase manipulation or direct rotation of planar filter. 
    Cyclic permutation of channels is performed for kernels on the group G.

    Args:
        w:
        filter_type:
        input_layer:
        nr_orients_out:
        basis_filters:
        rot_info:

    Returns:
        rot_filters:
    """

    if filter_type == 'steerable':
        # if using steerable filters, then rotate by phase manipulation

        rot_filters = [None] * nr_orients_out
        for orientation in range(nr_orients_out):
            rot_info_tmp = tf.expand_dims(rot_info[..., orientation], -1)
            filter_tmp = w * rot_info_tmp * basis_filters  # phase manipulation
            rot_filters[orientation] = filter_tmp
        # [nr_orients_out, J, K, K, nr_orients_in, filters_in, filters_out] (M: nr frequencies, R: nr radial profile params)
        rot_filters = tf.stack(rot_filters)

        # Linear combination of basis filters
        # [nr_orients_out, K, K, nr_orients_in, filters_in, filters_out]
        rot_filters = tf.reduce_sum(rot_filters, axis=1)
        # Get real part of filters
        # [nr_orients_out, K, K, nr_orients_in, filters_in, filters_out]
        rot_filters = tf.math.real(rot_filters, name='filters')

    else:
        # if using regular kernels, rotate by sparse matrix multiplication

        # [K, K, nr_orients_in, filters_in, filters_out]
        filter_shape = w.get_shape().as_list()

        # Flatten the filter
        filter_flat = tf.reshape(
            w, [filter_shape[0]*filter_shape[1], filter_shape[2]*filter_shape[3]*filter_shape[4]])

        # Generate a set of rotated kernels via rotation matrix multiplication
        idx, vals = MultiRotationOperatorMatrixSparse(
            [filter_shape[0], filter_shape[1]], nr_orients_out, periodicity=2*np.pi, diskMask=True)

        # Sparse rotation matrix
        rotOp_matrix = tf.SparseTensor(
            idx, vals, [nr_orients_out*filter_shape[0]*filter_shape[1], filter_shape[0]*filter_shape[1]])

        # Matrix multiplication
        rot_filters = tf.sparse_tensor_dense_matmul(
            rotOp_matrix, filter_flat)
        #[nr_orients_out * K * K, filters_in * filters_out]

        # Reshape the filters to [nr_orients_out, K, K, nr_orients_in, filters_in, filters_out]
        rot_filters = tf.reshape(
            rot_filters, [nr_orients_out, filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3], filter_shape[4]])

    # Do not cycle filter for input convolution f: Z2 -> G
    if input_layer is False:
        shape_list = rot_filters.get_shape().as_list()
        # cycle channels - [nr_orients_out, K, K, nr_orients_in, filters_in, filters_out]
        rot_filters = cycle_channels(rot_filters, shape_list)

    return rot_filters
####


def GConv2D(
        name,
        inputs,
        filters_out,
        kernel_size,
        nr_orients,
        filter_type,
        basis_filters=None,
        rot_info=None,
        input_layer=False,
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC',
        activation='bnrelu',
        use_bias=True,
        bias_initializer=tf.zeros_initializer()):
    """
    Rotation equivatiant group convolution layer

    Args:
        name: variable scope name
        inputs: input tensor
        filters_out:
        kernel_size: size of kernel
        basis_filters:
        rot_info:
        input_layer: whether the operation is the input layer (1st conv)
        strides: stride of kernel for convolution
        padding: choose either 'SAME' or 'VALID'
        data_format:
        activation: activation function to apply
        use_bias: whether to use bias
        bias_initializer: bias initialiser method

    Returns:
        conv: group equivariantconvolution of input with 
              steerable filters and optional activation.
    """

    if filter_type == 'steerable':
        assert basis_filters != None and rot_info != None, 'Must provide basis filters and rotation matrix'

    in_shape = inputs.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1

    if input_layer == False:
        nr_orients_in = nr_orients
    else:
        nr_orients_in = 1
    nr_orients_out = nr_orients

    filters_in = int(in_shape[channel_axis] / nr_orients_in)

    if filter_type == 'steerable':
        # shape for the filter coefficients
        nr_b_filts = basis_filters.shape[0]
        w_shape = [nr_b_filts, 1, 1, nr_orients_in, filters_in, filters_out]

        # init complex valued weights with the adapted He init (Weiler et al.)
        w1 = tf.get_variable(name + '_W_real', w_shape,
                             initializer=steerable_initializer(input_layer, nr_orients_out))
        w2 = tf.get_variable(name + '_W_imag', w_shape,
                             initializer=steerable_initializer(input_layer, nr_orients_out))
        w = tf.complex(w1, w2)

        # Generate filters at different orientations- also perform cyclic permutation of channels if f: G -> G
        # Cyclic permutation of filters happenens for all rotation equivariant layers except for the input layer
        # [nr_orients_out, K, K, nr_orients_in, filters_in, filters_out]
        filters = gen_rotated_filters(
            w, filter_type, input_layer, nr_orients_out, basis_filters, rot_info)

    else:
        w_shape = [kernel_size, kernel_size,
                   nr_orients_in, filters_in, filters_out]
        w = tf.get_variable(
            name + '_W', w_shape, initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))

        # Generate filters at different orientations- also perform cyclic permutation of channels if f: G -> G
        # Cyclic permutation of filters happenens for all rotation equivariant layers except for the input layer
        # [nr_orients_out, K, K, nr_orients_in, filters_in, filters_out]
        filters = gen_rotated_filters(
            w, filter_type, input_layer, nr_orients_out)

    # reshape filters for 2D convolution
    # [K, K, nr_orients_in, filters_in, nr_orients_out, filters_out]
    filters = tf.transpose(filters, [1, 2, 3, 4, 0, 5])
    filters = tf.reshape(filters, [
                         kernel_size, kernel_size, nr_orients_in * filters_in, nr_orients_out * filters_out])

    # perform conv with rotated filters (rehshaped so we can perform 2D convolution)
    kwargs = dict(data_format=data_format)
    conv = tf.nn.conv2d(inputs, filters, strides, padding.upper(), **kwargs)
    if use_bias:
        # Use same bias for all orientations
        b = tf.get_variable(
            name + '_bias', [filters_out], initializer=tf.zeros_initializer())
        b = tf.stack([b] * nr_orients_out)
        b = tf.reshape(b, [nr_orients_out*filters_out])
        conv = tf.nn.bias_add(conv, b)

    if activation == 'bnrelu':
        # Rotation equivariant batch normalisation
        conv = GBNReLU(name, conv, nr_orients_out)

    if activation == 'bn':
        # Rotation equivariant batch normalisation
        conv = GBatchNorm(name, conv, nr_orients_out)

    if activation == 'relu':
        # Rotation equivariant batch normalisation
        conv = tf.nn.relu(conv)

    return conv
