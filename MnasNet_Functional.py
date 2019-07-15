#!/usr/bin/python
# -*- coding: utf-8 -*-
import collections
import numpy as np
import tensorflow as tf
import re

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size',
    'num_repeat',
    'input_filters',
    'output_filters',
    'expand_ratio',
    'id_skip',
    'strides',
    'se_ratio',
    ])

# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None, ) * len(BlockArgs._fields)


GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum',
    'batch_norm_epsilon',
    'dropout_rate',
    'data_format',
    'input_shape',
    'num_classes',
    'depth_multiplier',
    'depth_divisor',
    'min_depth',
    'stem_size',
    'normalize_input'
    ])
GlobalParams.__new__.__defaults__ = (None, ) * len(GlobalParams._fields)


# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]



def decode_block_string(block_string):
    """Gets a MNasNet block through a string notation of arguments.
    E.g. r2_k3_s2_e1_i32_o16_se0.25_noskip: r - number of repeat blocks,
    k - kernel size, s - strides (1-9), e - expansion ratio, i - input filters,
    o - output filters, se - squeeze/excitation ratio
    Args:
      block_string: a string, a string representation of block arguments.
    Returns:
      A BlockArgs instance.
    Raises:
      ValueError: if the strides option is not correctly specified.
    """

    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
        splits = re.split(r'(\d.*)', op)
        if len(splits) >= 2:
            (key, value) = splits[:2]
            options[key] = value

    if 's' not in options or len(options['s']) != 2:
        raise ValueError('Strides options should be a pair of integers.')

    return BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip='noskip' not in block_string,
        se_ratio=(float(options['se']) if 'se' in options else None),
        strides=[int(options['s'][0]), int(options['s'][1])],
        )


def conv_kernel_initializer(shape, dtype=None):
    """Initialization for convolutional kernels.
  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
  a corrected standard deviation.
  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused
  Returns:
    an initialization for the variable
  """

    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None):
    """Initialization for dense kernels.
  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.
  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused
  Returns:
    an initialization for the variable
  """

    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random.uniform(shape, -init_range, init_range,
                             dtype=dtype)


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""

    multiplier = global_params.depth_multiplier
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters


def MnasBlock(input_tensor, block_args, global_params, name):

    batch_norm_momentum = global_params.batch_norm_momentum
    batch_norm_epsilon = global_params.batch_norm_epsilon
    data_format = global_params.data_format

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]
    has_se = block_args.se_ratio is not None and (block_args.se_ratio > 0
        and block_args.se_ratio <= 1)

    x = input_tensor
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:

        # Expansion phase:
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False,
            data_format=data_format,
            name=name + '_expand_conv',
            )(input_tensor)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon, fused=True,
                name=name+'_expand_conv_BN')(x)
        x = tf.keras.layers.ReLU()(x)

    kernel_size = block_args.kernel_size
    
    # Depth-wise convolution phase:
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=[kernel_size, kernel_size],
        strides=block_args.strides,
        depthwise_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        data_format=data_format,
        name=name+'_depthwise_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis,
            momentum=batch_norm_momentum, epsilon=batch_norm_epsilon,
            fused=True, name=name+'_depthwise_conv_BN')(x)
    x = tf.keras.layers.ReLU()(x)

    if has_se:
        num_reduced_filters = max(1, int(block_args.input_filters
                                         * block_args.se_ratio))

       # Squeeze and Excitation layer.
        se_tensor = tf.reduce_mean(x, spatial_dims, keepdims=True)
        se_tensor = tf.keras.layers.Conv2D(
            filters=num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=True,
            data_format=data_format,
            name=name +'_se_reduce_conv',
            )(se_tensor)
        se_tensor = tf.keras.layers.ReLU()(se_tensor)
        se_tensor = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=True,
            data_format=data_format,
            name=name+'_se_expand_conv',
            )(se_tensor)
        x = tf.sigmoid(se_tensor) * x

    # Output phase:
    filters = block_args.output_filters
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        data_format=data_format,
        name=name+'_project_conv',
        )(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis,
            momentum=batch_norm_momentum, epsilon=batch_norm_epsilon,
            fused=True, name=name+'_project_conv_BN')(x)

    if block_args.id_skip:
        if all(s == 1 for s in block_args.strides
        ) and block_args.input_filters == block_args.output_filters:
            x = tf.keras.layers.add([x, input_tensor], name=name+'_add')
    return x


def MnasNetModel(blocks_args, global_params):

    batch_norm_momentum = global_params.batch_norm_momentum
    batch_norm_epsilon = global_params.batch_norm_epsilon
    channel_axis = (1 if global_params.data_format == 'channels_first'
                     else -1)
    stem_size = global_params.stem_size
    data_format = global_params.data_format

    if data_format == 'channels_first':
        stats_shape = [3, 1, 1]
    else:
        stats_shape = [1, 1, 3]

    # Process input
    input_tensor = tf.keras.layers.Input(shape=global_params.input_shape,
                                         name='float_image_input')
    # Normalize the image to zero mean and unit variance.
    x = input_tensor
    if global_params.normalize_input:
        x -= tf.constant(MEAN_RGB, shape=stats_shape)
        x /= tf.constant(STDDEV_RGB, shape=stats_shape)
        

    # Stem part.
    x = tf.keras.layers.Conv2D(
        filters=round_filters(stem_size, global_params),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        data_format=data_format,
        name='stem_conv',
        )(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis,
            momentum=batch_norm_momentum, epsilon=batch_norm_epsilon,
            fused=True, name='stem_conv_BN')(x)
    x = tf.keras.layers.ReLU()(x)

    # Builds blocks.
    for (i, block_args) in enumerate(blocks_args):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args = \
            block_args._replace(input_filters=round_filters(block_args.input_filters,
                                global_params),
                                output_filters=round_filters(block_args.output_filters,
                                global_params))

        # The first block needs to take care of stride and filter size increase.
        name = 'block_{}__num{}_'.format(i, 0)
        x = MnasBlock(x, block_args, global_params, name)

        if block_args.num_repeat > 1:
            block_args = block_args._replace(input_filters=block_args.output_filters,
                                             strides=[1, 1])
        for j in range(1, block_args.num_repeat):
            name = 'block_{}__num{}_'.format(i, j)
            x = MnasBlock(x, block_args, global_params, name)

    # Head part.
    x = tf.keras.layers.Conv2D(
        filters=1280,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        data_format=data_format,
        name='head_conv',
        )(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis,
            momentum=batch_norm_momentum, epsilon=batch_norm_epsilon,
            fused=True, name='head_conv_BN')(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format,
            name='avg_pooling')(x)

    if global_params.dropout_rate > 0:
        x = tf.keras.layers.Dropout(global_params.dropout_rate)(x)

    output_fc = tf.keras.layers.Dense(
        global_params.num_classes,
        kernel_initializer=dense_kernel_initializer,
        name='fully_connected')(x)

    output_softmax = tf.keras.layers.Softmax(name='softmax')(output_fc)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_softmax)
    return model


def mnasnet_b1(depth_multiplier=None):
    """Creates a mnasnet-b1 model.
  Args:
    depth_multiplier: multiplier to number of filters per layer.
  Returns:
    blocks_args: a list of BlocksArgs for internal MnasNet blocks.
    global_params: GlobalParams, global parameters for the model.
  """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_noskip',
        'r3_k3_s22_e3_i16_o24',
        'r3_k5_s22_e3_i24_o40',
        'r3_k5_s22_e6_i40_o80',
        'r2_k3_s11_e6_i80_o96',
        'r4_k5_s22_e6_i96_o192',
        'r1_k3_s11_e6_i192_o320_noskip',
        ]
    decoded_blocks = [decode_block_string(s) for s in blocks_args]
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=0.2,
        data_format='channels_last',
        input_shape=(224, 224, 3),
        num_classes=1000,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        stem_size=32,
        normalize_input=True
        )
    return (decoded_blocks, global_params)


def mnasnet_a1(depth_multiplier=None):
    """Creates a mnasnet-a1 model.
  Args:
    depth_multiplier: multiplier to number of filters per layer.
  Returns:
    blocks_args: a list of BlocksArgs for internal MnasNet blocks.
    global_params: GlobalParams, global parameters for the model.
  """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_noskip',
        'r2_k3_s22_e6_i16_o24',
        'r3_k5_s22_e3_i24_o40_se0.25',
        'r4_k3_s22_e6_i40_o80',
        'r2_k3_s11_e6_i80_o112_se0.25',
        'r3_k5_s22_e6_i112_o160_se0.25',
        'r1_k3_s11_e6_i160_o320',
        ]
    decoded_blocks = [decode_block_string(s) for s in blocks_args]
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=0.2,
        data_format='channels_last',
        input_shape=(224, 224, 3),
        num_classes=1000,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None,
        stem_size=32,
        normalize_input=True
        )

    return (decoded_blocks, global_params)


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model."""

    if model_name == 'mnasnet-a1':
        (blocks_args, global_params) = mnasnet_a1()
    elif model_name == 'mnasnet-b1':
        (blocks_args, global_params) = mnasnet_b1()
    else:

  # elif model_name == 'mnasnet-small':
  #   blocks_args, global_params = mnasnet_small()
  # elif model_name == 'mnasnet-d1':
  #   blocks_args, global_params = mnasnet_d1()
  # elif model_name == 'mnasnet-d1-320':
  #   blocks_args, global_params = mnasnet_d1_320()

        raise NotImplementedError('model name is not pre-defined: %s'
                                  % model_name)

    if override_params:

    # ValueError will be raised here if override_params has fields not included
    # in global_params.

        global_params = global_params._replace(**override_params)
    return (blocks_args, global_params)


def build_mnasnet_model(model_name, override_params=None):
    """A helper functiion to create a MnasNet model and return predicted logits.
  Args:
    images: input images tensor.
    model_name: string, the model name of a pre-defined MnasNet.
    training: boolean, whether the model is constructed for training.
    override_params: A dictionary of params for overriding. Fields must exist in
      mnasnet_model.GlobalParams.
  Returns:
    logits: the logits tensor of classes.
    endpoints: the endpoints for each layer.
  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """

    (blocks_args, global_params) = get_model_params(model_name,
            override_params)
    return MnasNetModel(blocks_args, global_params)


if __name__ == '__main__':

    # tf.enable_eager_execution()

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_noskip',
        'r2_k3_s22_e6_i16_o24',
        'r3_k5_s22_e3_i24_o40_se0.25',
        'r4_k3_s22_e6_i40_o80',
        'r2_k3_s11_e6_i80_o112_se0.25',
        'r3_k5_s22_e6_i112_o160_se0.25',
        'r1_k3_s11_e6_i160_o320',
        ]
    decoded_strings = [decode_block_string(s) for s in blocks_args]
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=0.2,
        data_format='channels_last',
        num_classes=1000,
        depth_multiplier=None,
        depth_divisor=8,
        min_depth=None,
        stem_size=32,
        use_keras=True,
        )

    inp = tf.keras.layers.Input(shape=(224, 224, 3))
    x = MnasNetModel(inp, decoded_strings, global_params)
    model = tf.keras.models.Model(inputs=inp, outputs=x)
    print(model.summary())