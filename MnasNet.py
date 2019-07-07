import collections
import numpy as np
import tensorflow as tf
import re




BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'depth_multiplier', 'depth_divisor', 'min_depth',
    'stem_size', 'use_keras'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)


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
        key, value = splits[:2]
        options[key] = value

    if 's' not in options or len(options['s']) != 2:
      raise ValueError('Strides options should be a pair of integers.')

    return BlockArgs(
    	kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]), int(options['s'][1])])




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
  return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


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




class MnasBlock(object):
  """A class of MnasNet Inveretd Residual Bottleneck.
  Attributes:
    has_se: boolean. Whether the block contains a Squeeze and Excitation layer
      inside.
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, block_args, global_params):
    """Initializes a MnasNet block.
    Args:
      block_args: BlockArgs, arguments to create a MnasBlock.
      global_params: GlobalParams, a set of global parameters.
    """
    self._block_args = block_args
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    self._data_format = global_params.data_format
    if self._data_format == 'channels_first':
      self._channel_axis = 1
      self._spatial_dims = [2, 3]
    else:
      self._channel_axis = -1
      self._spatial_dims = [1, 2]
    self.has_se = (self._block_args.se_ratio is not None) and (
        self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)

    self.endpoints = None

    # Builds the block accordings to arguments.
    self._build()


  def block_args(self):
    return self._block_args


  def _build(self):
    """Builds MnasNet block according to the arguments."""
    filters = self._block_args.input_filters * self._block_args.expand_ratio
    if self._block_args.expand_ratio != 1:
      # Expansion phase:
      self._expand_conv = tf.keras.layers.Conv2D(
      	filters=filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        data_format=self._data_format)

      # TODO(hongkuny): b/120622234 need to manage update ops directly.
      self._expand_conv_BN = tf.keras.layers.BatchNormalization(
          axis=self._channel_axis,
          momentum=self._batch_norm_momentum,
          epsilon=self._batch_norm_epsilon,
          fused=True)

    kernel_size = self._block_args.kernel_size
    # Depth-wise convolution phase:
    self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
    	kernel_size=[kernel_size, kernel_size],
        strides=self._block_args.strides,
        depthwise_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        data_format=self._data_format)
   
    self._depthwise_conv_BN = tf.keras.layers.BatchNormalization(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        fused=True)

    if self.has_se:
      num_reduced_filters = max(
          1, int(self._block_args.input_filters * self._block_args.se_ratio))
      # Squeeze and Excitation layer.
      self._se_reduce = tf.keras.layers.Conv2D(
          filters=num_reduced_filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=True,
          data_format=self._data_format)
      self._se_expand = tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=True,
          data_format=self._data_format)

    # Output phase:
    filters = self._block_args.output_filters
    self._project_conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        data_format=self._data_format)
    self._project_conv_BN = tf.keras.layers.BatchNormalization(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon,
        fused=True)


  def _call_se(self, input_tensor):
    """Call Squeeze and Excitation layer.
    Args:
      input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.
    Returns:
      A output tensor, which should have the same shape as input.
    """
    se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keepdims=True)
    se_tensor = self._se_expand(tf.nn.relu(self._se_reduce(se_tensor)))
    tf.compat.v1.logging.info('Built Squeeze and Excitation with tensor shape: %s' %
                    (se_tensor.shape))
    return tf.sigmoid(se_tensor) * input_tensor


  def call(self, inputs, training=True):
    """Implementation of MnasBlock call().
    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
    Returns:
      A output tensor.
    """
    tf.compat.v1.logging.info('Block input: %s shape: %s' % (inputs.name, inputs.shape))
    if self._block_args.expand_ratio != 1:
      x = tf.nn.relu(self._expand_conv_BN(self._expand_conv(inputs), training=training))
    else:
      x = inputs
    tf.compat.v1.logging.info('Expand: %s shape: %s' % (x.name, x.shape))

    x = tf.nn.relu(self._depthwise_conv_BN(self._depthwise_conv(x), training=training))
    tf.compat.v1.logging.info('DWConv: %s shape: %s' % (x.name, x.shape))

    if self.has_se:
        x = self._call_se(x)

    self.endpoints = {'expansion_output': x}

    x = self._project_conv_BN(self._project_conv(x), training=training)
    if self._block_args.id_skip:
      if all(
          s == 1 for s in self._block_args.strides
      ) and self._block_args.input_filters == self._block_args.output_filters:
        x = tf.add(x, inputs)
    tf.compat.v1.logging.info('Project: %s shape: %s' % (x.name, x.shape))
    return x


class MnasNetModel(tf.keras.Model):
  """A class implements tf.keras.Model for MnesNet model.
    Reference: https://arxiv.org/abs/1807.11626
  """

  def __init__(self, blocks_args=None, global_params=None):
    """Initializes an `MnasNetModel` instance.
    Args:
      blocks_args: A list of BlockArgs to construct MnasNet block modules.
      global_params: GlobalParams, a set of global parameters.
    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super(MnasNetModel, self).__init__()
    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
    self._global_params = global_params
    self._blocks_args = blocks_args
    self.endpoints = None
    self._build()

  def _build(self):
    """Builds a MnasNet model."""
    self._blocks = []
    # Builds blocks.
    for block_args in self._blocks_args:
      assert block_args.num_repeat > 0
      # Update block input and output filters based on depth multiplier.
      block_args = block_args._replace(
          input_filters=round_filters(block_args.input_filters,
                                      self._global_params),
          output_filters=round_filters(block_args.output_filters,
                                       self._global_params))

      # The first block needs to take care of stride and filter size increase.
      self._blocks.append(MnasBlock(block_args, self._global_params))
      if block_args.num_repeat > 1:
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in range(block_args.num_repeat - 1):
        self._blocks.append(MnasBlock(block_args, self._global_params))

    batch_norm_momentum = self._global_params.batch_norm_momentum
    batch_norm_epsilon = self._global_params.batch_norm_epsilon
    channel_axis = 1 if self._global_params.data_format == 'channels_first' else -1

    # Stem part.
    stem_size = self._global_params.stem_size
    self._conv_stem = tf.keras.layers.Conv2D(
        filters=round_filters(stem_size, self._global_params),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        data_format=self._global_params.data_format)
    self._conv_stem_BN = tf.keras.layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        fused=True)

    # Head part.
    self._conv_head = tf.keras.layers.Conv2D(
        filters=1280,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        data_format=self._global_params.data_format)
    self._conv_head_BN = tf.keras.layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        fused=True)

    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
        data_format=self._global_params.data_format)
    self._fc = tf.keras.layers.Dense(
    	self._global_params.num_classes,
        kernel_initializer=dense_kernel_initializer)
    
    if self._global_params.dropout_rate > 0:
      self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)
    else:
      self._dropout = None


  def call(self, inputs, training=True, features_only=None):
    """Implementation of MnasNetModel call().
    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.
      features_only: build the base feature network only.
    Returns:
      output tensors.
    """
    outputs = None
    self.endpoints = {}
    # Calls Stem layers
    outputs = tf.nn.relu(
    	self._conv_stem_BN(self._conv_stem(inputs), training=training))
    tf.compat.v1.logging.info('Built stem layers with output shape: %s' % outputs.shape)
    self.endpoints['stem'] = outputs

    # Calls blocks.
    reduction_idx = 0
    for idx, block in enumerate(self._blocks):
      is_reduction = False
      if ((idx == len(self._blocks) - 1) or
          self._blocks[idx + 1].block_args().strides[0] > 1):
        is_reduction = True
        reduction_idx += 1

      outputs = block.call(outputs, training=training)
      self.endpoints['block_%s' % idx] = outputs
      if is_reduction:
        self.endpoints['reduction_%s' % reduction_idx] = outputs
      if block.endpoints:
        for k, v in block.endpoints.items():
          self.endpoints['block_%s/%s' % (idx, k)] = v
          if is_reduction:
            self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
    self.endpoints['global_pool'] = outputs

    if not features_only:
      # Calls final layers and returns logits.
      outputs = tf.nn.relu(
          self._conv_head_BN(self._conv_head(outputs), training=training))
      outputs = self._avg_pooling(outputs)
      if self._dropout:
        outputs = self._dropout(outputs, training=training)
      outputs = self._fc(outputs)
      self.endpoints['head'] = outputs
    return outputs


def mnasnet_b1(depth_multiplier=None):
  """Creates a mnasnet-b1 model.
  Args:
    depth_multiplier: multiplier to number of filters per layer.
  Returns:
    blocks_args: a list of BlocksArgs for internal MnasNet blocks.
    global_params: GlobalParams, global parameters for the model.
  """
  blocks_args = [
      'r1_k3_s11_e1_i32_o16_noskip', 'r3_k3_s22_e3_i16_o24',
      'r3_k5_s22_e3_i24_o40', 'r3_k5_s22_e6_i40_o80', 'r2_k3_s11_e6_i80_o96',
      'r4_k5_s22_e6_i96_o192', 'r1_k3_s11_e6_i192_o320_noskip'
  ]
  decoded_strings = [decode_block_string(s) for s in blocks_args]
  global_params = GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=0.2,
      data_format='channels_last',
      num_classes=1000,
      depth_multiplier=depth_multiplier,
      depth_divisor=8,
      min_depth=None,
      stem_size=32,
      use_keras=True)
  return decoded_strings, global_params


def mnasnet_a1(depth_multiplier=None):
  """Creates a mnasnet-a1 model.
  Args:
    depth_multiplier: multiplier to number of filters per layer.
  Returns:
    blocks_args: a list of BlocksArgs for internal MnasNet blocks.
    global_params: GlobalParams, global parameters for the model.
  """
  blocks_args = [
      'r1_k3_s11_e1_i32_o16_noskip', 'r2_k3_s22_e6_i16_o24',
      'r3_k5_s22_e3_i24_o40_se0.25', 'r4_k3_s22_e6_i40_o80',
      'r2_k3_s11_e6_i80_o112_se0.25', 'r3_k5_s22_e6_i112_o160_se0.25',
      'r1_k3_s11_e6_i160_o320'
  ]
  decoded_strings = [decode_block_string(s) for s in blocks_args]
  global_params = GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=0.2,
      data_format='channels_last',
      num_classes=1000,
      depth_multiplier=depth_multiplier,
      depth_divisor=8,
      min_depth=None,
      stem_size=32,
      use_keras=True)

  return decoded_strings, global_params



def get_model_params(model_name, override_params):
  """Get the block args and global params for a given model."""
  if model_name == 'mnasnet-a1':
    blocks_args, global_params = mnasnet_a1()
  elif model_name == 'mnasnet-b1':
    blocks_args, global_params = mnasnet_b1()
  # elif model_name == 'mnasnet-small':
  #   blocks_args, global_params = mnasnet_small()
  # elif model_name == 'mnasnet-d1':
  #   blocks_args, global_params = mnasnet_d1()
  # elif model_name == 'mnasnet-d1-320':
  #   blocks_args, global_params = mnasnet_d1_320()
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)

  if override_params:
    # ValueError will be raised here if override_params has fields not included
    # in global_params.
    global_params = global_params._replace(**override_params)
  return blocks_args, global_params


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
  blocks_args, global_params = get_model_params(model_name, override_params)
  return MnasNetModel(blocks_args, global_params)

import inspect

if __name__ == "__main__":
    # tf.enable_eager_execution()
    model = build_mnasnet_model('mnasnet-a1')
    print(model)
    print(type(model))
    model.build((None, 224, 224, 3))
    print(model.summary())

