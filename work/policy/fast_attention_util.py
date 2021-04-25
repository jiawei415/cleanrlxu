# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras-based einsum layer.

Copied from
https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/dense_einsum.py.
"""
# pylint: disable=g-classes-have-attributes

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
_CHR_IDX = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# @tf.keras.utils.register_keras_serializable(package="Text")
# class DenseEinsum(tf.keras.layers.Layer):
class DenseEinsum(nn.Module):
  """A densely connected layer that uses tf.einsum as the backing computation.

  This layer can perform einsum calculations of arbitrary dimensionality.

  Arguments:
    output_shape: Positive integer or tuple, dimensionality of the output space.
    num_summed_dimensions: The number of dimensions to sum over. Standard 2D
      matmul should use 1, 3D matmul should use 2, and so forth.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation")..
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix.
    bias_constraint: Constraint function applied to the bias vector.
  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`. The most common
      situation would be a 2D input with shape `(batch_size, input_dim)`.
  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`. For instance, for a 2D
      input with shape `(batch_size, input_dim)`, the output would have shape
      `(batch_size, units)`.
  """
  def __init__(self,
               input_shape,
               output_shape,
               num_summed_dimensions=1,
               activation=None,
               limit=0,
               use_bias=True,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(DenseEinsum, self).__init__(**kwargs)
    self._input_shape = input_shape if isinstance(
        input_shape, (list, tuple)) else (input_shape,)
    self._output_shape = output_shape if isinstance(
        output_shape, (list, tuple)) else (output_shape,)
    # self._activation = tf.keras.activations.get(activation)
    self._limit = limit
    self._use_bias = use_bias
    # self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    # self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    # self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    # self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    # self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    # self._bias_constraint = tf.keras.constraints.get(bias_constraint)
    self._num_summed_dimensions = num_summed_dimensions
    self._einsum_string = None

    free_input_dims = 2 # input_dims - self._num_summed_dimensions
    output_dims = len(self._output_shape)
    self._einsum_string = self._build_einsum_string(free_input_dims,
                                                    self._num_summed_dimensions,
                                                    output_dims)
    # This is only saved for testing purposes.
    # self._kernel_shape = input_shape[free_input_dims:] + self._output_shape
    self._kernel_shape = self._input_shape + self._output_shape
    _kernel = Parameter(torch.Tensor(torch.Size(self._kernel_shape)))
    self._kernel = nn.init.uniform_(_kernel, a=-self._limit, b=self._limit) # .to(self.device)
    # self._kernel = self.add_weight(
    #     "kernel",
    #     shape=self._kernel_shape,
    #     initializer=self._kernel_initializer,
    #     regularizer=self._kernel_regularizer,
    #     constraint=self._kernel_constraint,
    #     dtype=self.dtype,
    #     trainable=True)
    if self._use_bias:
      # self._bias = self.add_weight(
      #     "bias",
      #     shape=self._output_shape,
      #     initializer=self._bias_initializer,
      #     regularizer=self._bias_regularizer,
      #     constraint=self._bias_constraint,
      #     dtype=self.dtype,
      #     trainable=True)
      _bias = Parameter(torch.Size(self._output_shape))
      self._bias = nn.init.zeros_(_bias)
    else:
      self._bias = None


  def extra_repr(self):
    return 'kernel={}, bias={}'.format(
      self._kernel_shape, self._use_bias
    )


  def _build_einsum_string(self, free_input_dims, bound_dims, output_dims):
    input_str = ""
    kernel_str = ""
    output_str = ""
    letter_offset = 0
    for i in range(free_input_dims):
      char = _CHR_IDX[i + letter_offset]
      input_str += char
      output_str += char

    letter_offset += free_input_dims
    for i in range(bound_dims):
      char = _CHR_IDX[i + letter_offset]
      input_str += char
      kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
      char = _CHR_IDX[i + letter_offset]
      kernel_str += char
      output_str += char

    return input_str + "," + kernel_str + "->" + output_str

  def forward(self, inputs):
    ret = torch.einsum(self._einsum_string, inputs, self._kernel)
    if self._use_bias:
      ret += self._bias
    # if self._activation is not None:
    #   ret = self._activation(ret)
    return ret

