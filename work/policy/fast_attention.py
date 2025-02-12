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

"""Implementation of multiheaded FAVOR-attention & FAVOR-self-attention layers.

Prefix Sum Tensorflow implementation by Valerii Likhosherstov.
"""
import math
import fast_attention_util as util
import torch
import torch.nn as nn
import torch.nn.functional as F


BIG_CONSTANT = 1e8


def create_projection_matrix(m, d, seed=0, scaling=0):
  r"""Constructs the matrix of random projections.

  Constructs a matrix of random orthogonal projections. Each projection vector
  has direction chosen uniformly at random and either deterministic length
  \sqrt{d} or length taken from the \chi(d) distribution (in the latter case
  marginal distributions of the projections are d-dimensional Gaussian vectors
  with associated identity covariance matrix).

  Args:
    m: number of random projections.
    d: dimensionality of each random projection.
    seed: random seed used to construct projections.
    scaling: 1 if all the random projections need to be renormalized to have
      length \sqrt{d}, 0 if the lengths of random projections should follow
      \chi(d) distribution.

  Returns:
    The matrix of random projections of the shape [m, d].
  """
  nb_full_blocks = int(m / d)
  block_list = []
  current_seed = seed
  for _ in range(nb_full_blocks):
    torch.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)
    unstructured_block = torch.normal(size=(d, d))
    q, _ = torch.qr(unstructured_block)
    q = torch.transpose(q)
    block_list.append(q)
    current_seed += 1
  remaining_rows = m - nb_full_blocks * d
  if remaining_rows > 0:
    unstructured_block = torch.normal(size=(d, d))
    q, _ = torch.qr(unstructured_block)
    q = torch.transpose(q)
    block_list.append(q[0:remaining_rows])
  final_matrix = torch.stack(block_list, 1)

  current_seed += 1
  torch.manual_seed(current_seed)
  torch.cuda.manual_seed_all(current_seed)

  if scaling == 0:
    multiplier = torch.norm(torch.normal(size=(m, d)), dim=1)
  elif scaling == 1:
    multiplier = math.sqrt(float(d)) * torch.ones(m)
  else:
    raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

  return torch.matmul(torch.diag(multiplier), final_matrix)


def relu_kernel_transformation(data,
                               is_query,
                               projection_matrix=None,
                               numerical_stabilizer=0.001):
  """Computes features for the ReLU-kernel.

  Computes random features for the ReLU kernel from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  del is_query
  if projection_matrix is None:
    return nn.ReLU()(data) + numerical_stabilizer
  else:

    ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
    data_dash = ratio * torch.einsum("blhd,md->blhm", data, projection_matrix)
    return nn.ReLU()(data_dash) + numerical_stabilizer


def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix=None,
                                  numerical_stabilizer=0.000001):
  """Computes random features for the softmax kernel using FAVOR+ mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  data_normalizer = 1.0 / (math.sqrt(math.sqrt(data.shape[-1])))
  ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
  data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)
  diag_data = data.pow(2)
  diag_data = torch.sum(diag_data, dim=data.dim() - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = diag_data.unsqueeze(dim=data.dim() - 1)
  if is_query:
    last_dims_t = len(data_dash.shape) - 1
    data_dash = ratio * (
      torch.exp(data_dash - diag_data - torch.max(
        data_dash, dim=last_dims_t, keepdim=True
      )[0]) + numerical_stabilizer
    )
  else:
    data_dash = ratio * (
      torch.exp(data_dash - diag_data - torch.max(data_dash)[0]) + numerical_stabilizer
    )
  return data_dash


def noncausal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR noncausal attention AV.
  """
  kvs = torch.einsum("lbhm,lbhd->bhmd", ks, vs)
  return torch.einsum("lbhm,bhmd->lbhd", qs, kvs)


def noncausal_denominator(qs, ks):
  """Computes FAVOR normalizer in noncausal attention.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in noncausal attention.
  """
  all_ones = torch.ones([ks.shape[0]]).to(ks.device)
  ks_sum = torch.einsum("lbhm,l->bhm", ks, all_ones)
  return torch.einsum("lbhm,bhm->lbh", qs, ks_sum)


def favor_attention(query,
                    key,
                    value,
                    kernel_transformation,
                    causal,
                    projection_matrix=None):
  """Computes FAVOR normalized attention.

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    kernel_transformation: transformation used to get finite kernel features.
    causal: whether attention is causal or not.
    projection_matrix: projection matrix to be used.

  Returns:
    FAVOR normalized attention.
  """
  query_prime = kernel_transformation(query, True,
                                      projection_matrix)  # [B,L,H,M]
  key_prime = kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]
  query_prime = query_prime.permute(1, 0, 2, 3)  # [L,B,H,M]
  key_prime = key_prime.permute(1, 0, 2, 3)  # [L,B,H,M]
  value = value.permute(1, 0, 2, 3)  # [L,B,H,D]

  if causal:
    av_attention = causal_numerator(query_prime, key_prime, value)
    attention_normalizer = causal_denominator(query_prime, key_prime)
  else:
    av_attention = noncausal_numerator(query_prime, key_prime, value)
    attention_normalizer = noncausal_denominator(query_prime, key_prime)
  # TODO(kchoro): Add more comments.
  av_attention = av_attention.permute(1, 0, 2, 3)
  attention_normalizer = attention_normalizer.permute(1, 0, 2)
  attention_normalizer = attention_normalizer.unsqueeze(dim=len(attention_normalizer.shape))

  return av_attention / attention_normalizer


class Attention(nn.Module):
  """Multi-headed attention layer."""
  def __init__(self,
               hidden_size,
               num_heads,
               attention_dropout=0.1,
               kernel_transformation=relu_kernel_transformation,
               numerical_stabilizer=0.001,
               causal=False,
               projection_matrix_type=None,
               nb_random_features=0):
    """Initialize Attention.

    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
      kernel_transformation: transformation used to produce kernel features for
        attention.
      numerical_stabilizer: used to bound away from zero kernel values.
      causal: whether attention is causal or not.
      projection_matrix_type: None if Identity should be used, otherwise random
        projection matrix will be applied.
      nb_random_features: number of random features to be used (relevant only if
        projection_matrix is not None).
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.kernel_transformation = kernel_transformation
    self.numerical_stabilizer = numerical_stabilizer
    self.causal = causal
    self.projection_matrix_type = projection_matrix_type
    self.nb_random_features = nb_random_features

    """Builds the layer."""
    # Layers for linearly projecting the queries, keys, and values.
    size_per_head = self.hidden_size // self.num_heads

    # def _glorot_initializer(fan_in, fan_out):
    #     limit = math.sqrt(6.0 / (fan_in + fan_out))
    #     return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

    def initializer_limit(fan_in, fan_out):
        return  math.sqrt(6.0 / (fan_in + fan_out))

    # attention_initializer = _glorot_initializer(input_shape[-1], self.hidden_size)
    attention_limit = initializer_limit(self.hidden_size, self.hidden_size)
    self.query_dense_layer = util.DenseEinsum(
        input_shape=self.hidden_size,
        output_shape=(self.num_heads, size_per_head),
        limit=attention_limit,
        # kernel_initializer=attention_initializer,
        use_bias=False)
    self.key_dense_layer = util.DenseEinsum(
        input_shape=self.hidden_size,
        output_shape=(self.num_heads, size_per_head),
        limit=attention_limit,
        # kernel_initializer=attention_initializer,
        use_bias=False)
    self.value_dense_layer = util.DenseEinsum(
        input_shape=self.hidden_size,
        output_shape=(self.num_heads, size_per_head),
        limit=attention_limit,
        # kernel_initializer=attention_initializer,
        use_bias=False)
    # output_initializer = _glorot_initializer(self.hidden_size, self.hidden_size)
    output_limit = initializer_limit(self.hidden_size, self.hidden_size)
    self.output_dense_layer = util.DenseEinsum(
        input_shape=(self.num_heads, size_per_head),
        output_shape=self.hidden_size,
        num_summed_dimensions=2,
        limit=output_limit,
        # kernel_initializer=output_initializer,
        use_bias=False)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
    }

  def forward(self,
           query_input,
           source_input,
           training,
           cache=None,
           decode_loop_step=None):
    """Apply attention mechanism to query_input and source_input.

    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      source_input: A tensor with shape [batch_size, length_source,
        hidden_size].
      bias: A tensor with shape [batch_size, 1, length_query, length_source],
        the attention bias that will be added to the result of the dot product.
      training: A bool, whether in training mode or not.
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, heads, dim_per_head],
             "v": tensor with shape [batch_size, i, heads, dim_per_head]} where
               i is the current decoded length for non-padded decode, or max
               sequence length for padded decode.
      decode_loop_step: An integer, step number of the decoding loop. Used only
        for autoregressive inference on TPU.

    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
    """
    # Linearly project the query, key and value using different learned
    # projections. Splitting heads is automatically done during the linear
    # projections --> [batch_size, length, num_heads, dim_per_head].

    query = self.query_dense_layer(query_input)
    key = self.key_dense_layer(source_input)
    value = self.value_dense_layer(source_input)

    if self.projection_matrix_type is None:
      projection_matrix = None
    else:
      dim = query.shape[-1]
      seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT))
      seed = int(seed)
      projection_matrix = create_projection_matrix(
          self.nb_random_features, dim, seed=seed)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      if decode_loop_step is not None:
        cache_k_shape = cache["k"].shape
        indices = torch.reshape(
            F.one_hot(decode_loop_step, cache_k_shape[1], dtype=key.dtype),
            [1, cache_k_shape[1], 1, 1])
        key = cache["k"] + key * indices
        cache_v_shape = cache["v"].shape
        indices = torch.reshape(
            F.one_hot(decode_loop_step, cache_v_shape[1], dtype=value.dtype),
            [1, cache_v_shape[1], 1, 1])
        value = cache["v"] + value * indices
      else:
        key = torch.cat([cache["k"], key], dim=1)
        value = torch.cat([cache["v"], value], dim=1)

      # Update cache
      cache["k"] = key
      cache["v"] = value

    attention_output = favor_attention(query, key, value,
                                       self.kernel_transformation, self.causal,
                                       projection_matrix)
    attention_output = self.output_dense_layer(attention_output)
    return attention_output

class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self,
           q, k, v,
           training=True,
           cache=None,
           decode_loop_step=None):
    return super(SelfAttention, self).call(q, k,
                                           training, cache, decode_loop_step)
