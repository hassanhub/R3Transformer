import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization
from tensorflow.keras.layers import Dense, Dropout, ReLU, Embedding
import numpy as np
import logging, re

def assign_weight_from_h5(w, h5_handle):
    success = False

    h5_to_native = [
                [r"^(.*)/tf_t5with_lm_head_model", r"transformer"],
                [r"^shared/shared/(.*)", r"transformer/decoder/decoder_embeddings/\1"],

                [r"^(.*)/block_._([0-99]+)/layer_._0/SelfAttention/(.*)", r"\1/layer_\2/self_attention/\3"],
                [r"^(.*)/block_._([0-99]+)/layer_._0/layer_norm/(.*)", r"\1/layer_\2/self_attention/layer_norm/\3"],
                
                [r"^(.*)/encoder/block_._([0-99]+)/layer_._1/DenseReluDense/(.*)", r"\1/encoder/layer_\2/feed_forward/dense_relu_dense/\3"],
                [r"^(.*)/encoder/block_._([0-99]+)/layer_._1/layer_norm/(.*)", r"\1/encoder/layer_\2/feed_forward/layer_norm/\3"],

                [r"^(.*)/decoder/block_._([0-99]+)/layer_._1/EncDecAttention/(.*)", r"\1/decoder/layer_\2/cross_attention/\3"],
                [r"^(.*)/decoder/block_._([0-99]+)/layer_._1/layer_norm/(.*)", r"\1/decoder/layer_\2/cross_attention/layer_norm/\3"],
                [r"^(.*)/decoder/block_._([0-99]+)/layer_._2/DenseReluDense/(.*)", r"\1/decoder/layer_\2/feed_forward/dense_relu_dense/\3"],
                [r"^(.*)/decoder/block_._([0-99]+)/layer_._2/layer_norm/(.*)", r"\1/decoder/layer_\2/feed_forward/layer_norm/\3"],
    ]

    native_to_h5 = [
                [r"^transformer/(.*)/layer_([0-99]+)/(.*)", r"\1/tf_t5with_lm_head_model/\1/block_._\2/\3"],
                [r"^transformer/(.*)/final_layer_norm/(.*)", r"\1/tf_t5with_lm_head_model/\1/final_layer_norm/\2"],
                [r"^transformer/encoder/encoder_embeddings/(.*)", r"shared/tf_t5with_lm_head_model/shared/\1"],
                [r"^transformer/decoder/decoder_embeddings/(.*)", r"shared/tf_t5with_lm_head_model/shared/\1"],

                [r"^(.*)/self_attention/layer_norm/(.*)", r"\1/layer_._0/layer_norm/\2"],
                [r"^(.*)/cross_attention/layer_norm/(.*)", r"\1/layer_._1/layer_norm/\2"],

                [r"^(.*)/encoder/block_._([0-99]+)/feed_forward/layer_norm/(.*)", r"\1/encoder/block_._\2/layer_._1/layer_norm/\3"],
                [r"^(.*)/decoder/block_._([0-99]+)/feed_forward/layer_norm/(.*)", r"\1/decoder/block_._\2/layer_._2/layer_norm/\3"],

                [r"^(.*)/self_attention/(.*)", r"\1/layer_._0/SelfAttention/\2"],
                [r"^(.*)/cross_attention/(.*)", r"\1/layer_._1/EncDecAttention/\2"],

                [r"^(.*)/encoder/block_._([0-99]+)/feed_forward/dense_relu_dense/(.*)", r"\1/encoder/block_._\2/layer_._1/DenseReluDense/\3"],
                [r"^(.*)/decoder/block_._([0-99]+)/feed_forward/dense_relu_dense/(.*)", r"\1/decoder/block_._\2/layer_._2/DenseReluDense/\3"],
    ]           

    h5_name = w.name
    for source, dest in native_to_h5:
        h5_name = re.sub(source, dest, h5_name)

    if h5_name in h5_handle:
        weights = h5_handle[h5_name][...]
        if weights.shape != w.numpy().shape:
            success = False
        else:    
            assigned = w.assign(weights)
            if np.linalg.norm(assigned-weights) == 0:
                success = True
            
    return success

def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
    Args:
        initializer_range: float, initializer range for stddev.
    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

def get_shape(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def mse(x_1, x_2, axis=-1):
    return tf.reduce_mean(tf.square(x_1-x_2), axis=axis)

class VectorQuantizer(Layer):
    def __init__(self,
                 d_model,
                 vocab_size,
                 dropout_rate=0.1,
                 name='vector_quantizer',
                 **kwargs):
        super(VectorQuantizer,self).__init__(name=name, **kwargs)
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.layernorm = LayerNormalization(epsilon=1e-6, name='layer_norm')
        self.dropout = Dropout(dropout_rate)
        self.o = Dense(d_model, use_bias=False, name='o')

    def build(self, input_shape):
        self.dict = self.add_weight(shape=(self.vocab_size,self.d_model),
                                    initializer=tf.keras.initializers.get('glorot_normal'),
                                    trainable=True,
                                    dtype=tf.float32,
                                    name='embeddings_dictionary')
    def freeze_layer(self):
        self.trainable = False

    def _calc_word_id(self, inputs, training):
        query = tf.nn.l2_normalize(inputs, axis=-1)
        key = tf.nn.l2_normalize(self.dict, axis=-1)
        sim_ = tf.matmul(query, key, transpose_b=True) # (batch_size, seq_len, vocab_size)
        sim_ = self.dropout(sim_, training)
        word_idx = tf.argmax(sim_, axis=-1) # (batch_size, seq_len)
        word_sim = tf.reduce_max(sim_, axis=-1) # (batch_size, seq_len)

        return word_idx, word_sim

    def call(self,
             inputs,
             training=False):

        y = inputs # (batch_size, seq_len, d_model) 

        #find nearest neighbor
        output_ids, outputs_scores = self._calc_word_id(y, training) # (batch_size, seq_len)
        z = tf.gather(self.dict, output_ids) # (batch_size, seq_len, d_model)

        #Straight-through estimation
        output_embeddings = tf.stop_gradient(z - y) + y # (batch_size, seq_len, d_model)
        output_embeddings = self.layernorm(output_embeddings)
        output_embeddings = self.o(output_embeddings)
        
        if training:
            dictionary_loss = mse(x_1=tf.stop_gradient(y), 
                                  x_2=z, 
                                  axis=[1,2])
            commitment_loss = mse(x_1=y,
                                  x_2=tf.stop_gradient(z),
                                  axis=[1,2])
            loss =  dictionary_loss + 0.25 * commitment_loss # (batch_size,)

        else:
            loss = 0.

        outputs = {'output_ids': output_ids,
                   'output_scores': outputs_scores,
                   'output_embeddings': output_embeddings,
                   'loss': loss}
                   
        return outputs

class SpatioTemporalEmbeddings(Layer):
    """Construct the embeddings from spatio-temporal tokens.
    """

    def __init__(self,
                 hidden_size,
                 max_temporal_positions,
                 max_spatial_centers,
                 dropout_rate=None,
                 initializer_range=None,
                 layer_norm_epsilon=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_temporal_positions = max_temporal_positions
        self.max_spatial_centers = max_spatial_centers
        self.max_spatial_size = int(np.sqrt(max_spatial_centers)) # assuming that max_h = max_w = sqrt(num_spatial_positions)
        self.hidden_size = hidden_size
        self.dropout_rate = 0.1 if dropout_rate is None else dropout_rate
        self.initializer_range = hidden_size ** -0.5 if initializer_range is None else initializer_range
        self.layer_norm_epsilon = 1e-6 if layer_norm_epsilon is None else layer_norm_epsilon

        self.temporal_position_embeddings = Embedding(
                    self.max_temporal_positions,
                    self.hidden_size,
                    embeddings_initializer=get_initializer(self.initializer_range),
                    name="temporal_position_embeddings",
                    )
        self.spatial_center_embeddings = Embedding(
                    self.max_spatial_centers,
                    self.hidden_size,
                    embeddings_initializer=get_initializer(self.initializer_range),
                    name="spatial_center_embeddings",
                    )

        self.spatial_size_embeddings = Embedding(
                    self.max_spatial_centers,
                    self.hidden_size,
                    embeddings_initializer=get_initializer(self.initializer_range),
                    name="spatial_size_embeddings",
                    )


        self.layernorm = T5LayerNorm(epsilon=self.layer_norm_epsilon, name="layer_norm")
        self.dropout = Dropout(self.dropout_rate)

    def _get_center_ids(self,
                        spatial_ids):

        center_x = (spatial_ids[:,:,0] + spatial_ids[:,:,1]) * 0.5
        center_x = tf.math.floor(center_x * self.max_spatial_size)

        center_y = (spatial_ids[:,:,2] + spatial_ids[:,:,3]) * 0.5
        center_y = tf.math.floor(center_y * self.max_spatial_size)

        center_ids = center_y * self.max_spatial_size + center_x
        center_ids = tf.cast(center_ids, dtype=tf.int32)

        return center_ids

    def _get_size_ids(self,
                      spatial_ids):
        length_x = tf.abs(spatial_ids[:,:,1] - spatial_ids[:,:,0])
        length_y = tf.abs(spatial_ids[:,:,3] - spatial_ids[:,:,2])

        size_ids = length_y * self.max_spatial_size + length_x
        size_ids = tf.cast(size_ids, dtype=tf.int32)

        return size_ids

    def call(self, position_ids, training=False):
        """Get token embeddings of inputs.
        Args:
            inputs: inputs with shape [batch_size, length, 5]: where 5 = 1 + 4 ; 1 for frame id, 4 for spatial location with
            the following order: [x0, x1, y0, y1]
        Returns:
            outputs: output embedding tensor, float32 with shape [batch_size, length, embedding_size]
        """
        temporal_position_ids = tf.cast(position_ids[:,:,0], dtype=tf.int32)
        spatial_position_ids = position_ids[:,:,1:]
        
        spatial_center_ids = self._get_center_ids(spatial_position_ids)
        spatial_size_ids = self._get_size_ids(spatial_position_ids)

        temporal_position_embeddings = self.temporal_position_embeddings(temporal_position_ids)
        spatial_center_embeddings = self.spatial_center_embeddings(spatial_center_ids)
        spatial_size_embeddings = self.spatial_size_embeddings(spatial_size_ids)

        position_embeddings = temporal_position_embeddings + spatial_center_embeddings + spatial_size_embeddings
        position_embeddings = self.layernorm(position_embeddings)
        position_embeddings = self.dropout(position_embeddings, training=training)
        
        return position_embeddings

class TemporalEmbeddings(Layer):
    """Construct the embeddings from spatio-temporal tokens.
    """

    def __init__(self,
                 hidden_size,
                 max_temporal_positions,
                 dropout_rate=None,
                 initializer_range=None,
                 layer_norm_epsilon=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_temporal_positions = max_temporal_positions
        self.hidden_size = hidden_size
        self.dropout_rate = 0.1 if dropout_rate is None else dropout_rate
        self.initializer_range = hidden_size ** -0.5 if initializer_range is None else initializer_range
        self.layer_norm_epsilon = 1e-6 if layer_norm_epsilon is None else layer_norm_epsilon

        self.temporal_position_embeddings = Embedding(
                    self.max_temporal_positions,
                    self.hidden_size,
                    embeddings_initializer=get_initializer(self.initializer_range),
                    name="temporal_position_embeddings",
                    )

        self.layernorm = T5LayerNorm(epsilon=self.layer_norm_epsilon, name="layer_norm")
        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        """Get token embeddings of inputs.
        Args:
            inputs: inputs with shape [batch_size, seq_len, hidden_size]
        Returns:
            outputs: output embedding tensor, float32 with shape [batch_size, seq_len, hidden_size]
        """
        seq_len = get_shape(inputs)[1]
        temporal_position_ids = tf.range(seq_len)
        
        temporal_position_embeddings = self.temporal_position_embeddings(temporal_position_ids) # (seq_len, hidden_size)

        position_embeddings = self.layernorm(temporal_position_embeddings)
        position_embeddings = self.dropout(position_embeddings, training=training)
        position_embeddings = position_embeddings[tf.newaxis, :, :] # (1, seq_len, hidden_size)
        
        return position_embeddings

class RelativeRole(Layer):
    def __init__(self,
                 d_kv,
                 n_heads,
                 vocab_size,
                 dropout_rate,
                 name='r3',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.inner_dim = n_heads * d_kv
        self.d_kv = d_kv
        self.n_heads = n_heads

        self.wi = Dense(self.inner_dim, use_bias=False, name='wi')

        self.layernorm = LayerNormalization(epsilon=1e-6, name='layer_norm')
        self.dropout = Dropout(dropout_rate)
        
        self.vq = VectorQuantizer(d_model=d_kv,
                                  vocab_size=vocab_size,
                                  name='vector_quantizer')

    def _split_heads(self, x, bs):
        """ split heads and rearrange elements """
        """ output shape: (bs, seq_len, n_heads, d_kv) """
        return tf.reshape(x, (bs, -1, self.n_heads, self.d_kv)) # (bs, seq_len, n_heads, d_kv)

    def call(self,
             query,
             key,
             q_position_embeddings,
             k_position_embeddings,
             training):

        bs, qlen, dim = get_shape(query)
        _, klen, _ = get_shape(key)
        
        query += q_position_embeddings
        key += k_position_embeddings

        query = self.wi(tf.concat([query, tf.zeros_like(query)], axis=-1))[:,:,tf.newaxis,:] # (bs, qlen, 1, inner_dim)
        key = self.wi(tf.concat([tf.zeros_like(key), key], axis=-1))[:,tf.newaxis,:,:] # (bs, 1, klen, inner_dim)

        role_cont_rep = tf.reshape(query + key, [-1, qlen*klen, dim]) # (bs, qlen*klen, inner_dim)
        role_cont_rep = self._split_heads(role_cont_rep, bs) # (bs, qlen*klen, n_heads, dim_per_head)
        role_cont_rep = tf.reshape(role_cont_rep, (bs, -1, self.d_kv)) # (bs, qlen*klen*n_heads, d_kv)
        outputs_vq = self.vq(role_cont_rep, training) # (bs, qlen*klen*n_heads, dim_per_head)
        #outputs_vq = {'output_embeddings': inputs_vq, 'output_scores': tf.ones((bs, qlen*klen), dtype=tf.float32), 'loss': 0.}
        
        outputs_vq['output_embeddings'] = tf.reshape(outputs_vq['output_embeddings'], [bs, -1, self.n_heads, self.d_kv]) # (bs, qlen*klen, n_heads, dim_per_head)
        outputs_vq['output_embeddings'] = tf.reshape(outputs_vq['output_embeddings'], [bs, qlen, klen, self.n_heads, self.d_kv])  # (bs, qlen, klen, n_heads, dim_per_head)
        outputs_vq['output_scores'] = tf.reshape(outputs_vq['output_scores'], [bs, -1, self.n_heads, 1])
        outputs_vq['output_scores'] = tf.reshape(outputs_vq['output_scores'], [bs, qlen, klen, self.n_heads, 1])
        outputs_vq['output_ids'] = tf.reshape(outputs_vq['output_ids'], [bs, -1, self.n_heads, 1])
        outputs_vq['output_ids'] = tf.reshape(outputs_vq['output_ids'], [bs, qlen, klen, self.n_heads, 1])
        
        r3 = outputs_vq['output_embeddings']
        r3_scores = outputs_vq['output_scores']

        r3_weighted = r3 * r3_scores # (bs, qlen, klen, n_heads, dim_per_head) - weighted relative roles
        r3_weighted = tf.reduce_mean(r3_weighted, axis=2) # (bs, qlen, n_heads, dim_per_head) - output relative role
        r3_weighted = tf.transpose(r3_weighted, [0,2,1,3]) # (bs, n_heads, qlen, dim_per_head) - output relative role

        r3_weighted = self.layernorm(r3_weighted)
        r3_weighted = self.dropout(r3_weighted, training)

        return r3_weighted, outputs_vq

class AttentiveRelativeRole(Layer):
    def __init__(self,
                 d_kv,
                 n_heads,
                 vocab_size,
                 dropout_rate,
                 name='attentive_r3',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.inner_dim = n_heads * d_kv
        self.d_kv = d_kv
        self.n_heads = n_heads

        self.q = Dense(self.inner_dim, use_bias=False, name="q")
        self.k = Dense(self.inner_dim, use_bias=False, name="k")
        self.v = Dense(self.inner_dim, use_bias=False, name="v")

        self.layernorm = LayerNormalization(epsilon=1e-6, name='layer_norm')
        self.dropout = Dropout(dropout_rate)
        
        self.vq = VectorQuantizer(d_model=d_kv,
                                  vocab_size=vocab_size,
                                  name='vector_quantizer')

    def _split_heads(self, x, bs):
        """ split heads and rearrange elements """
        """ output shape: (bs, n_heads, seq_len, d_kv) """
        return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, self.d_kv)), perm=(0, 2, 1, 3))

    def call(self,
             query,
             key,
             value,
             mask,
             q_position_embeddings,
             kv_position_embeddings,
             training):

        bs, qlen, dim = get_shape(query)
        _, klen, _ = get_shape(key)

        query += q_position_embeddings
        key += kv_position_embeddings
        value += kv_position_embeddings

        q = self.q(query)  # (bs, qlen, inner_dim)
        k = self.k(key)  # (bs, klen, inner_dim)
        v = self.v(value)  # (bs, vlen, inner_dim)

        q = self._split_heads(q, bs) # (bs, n_heads, qlen, dim_per_head)
        k = self._split_heads(k, bs) # (bs, n_heads, klen, dim_per_head)
        v = self._split_heads(v, bs) # (bs, n_heads, vlen, dim_per_head)

        scores = tf.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)
        if mask is not None:
            scores += mask  # (bs, n_heads, qlen, klen)
        attention_weights = tf.nn.softmax(scores, axis=-1)  # (bs, n_heads, qlen, klen)
        attention_weights = self.dropout(attention_weights, training=training)  # (bs, n_heads, qlen, klen)
        
        role_states = tf.matmul(attention_weights, v)  # (bs, n_heads, qlen, dim_per_head)
        role_states = tf.reshape(role_states, [bs, -1, self.d_kv]) # (bs, n_heads*qlen, dim_per_head)
        
        outputs_vq = self.vq(role_states, training) # (bs, n_heads*qlen, dim_per_head)
        
        outputs_vq['output_embeddings'] = tf.reshape(outputs_vq['output_embeddings'], [bs, self.n_heads, -1, self.d_kv]) # (bs, n_heads, qlen, dim_per_head)
        outputs_vq['output_ids'] = tf.reshape(outputs_vq['output_ids'], [bs, self.n_heads, -1, 1]) # (bs, n_heads, qlen, 1)
        outputs_vq['output_scores'] = tf.reshape(outputs_vq['output_scores'], [bs, self.n_heads, -1, 1]) # (bs, n_heads, qlen, 1)

        quantized_roles = outputs_vq['output_embeddings']
        quantized_roles = self.layernorm(quantized_roles)
        quantized_roles = self.dropout(quantized_roles, training)

        return quantized_roles, outputs_vq

class Embeddings(Layer):
    """Construct token embeddings.
    """

    def __init__(self, 
                 vocab_size, 
                 hidden_size, 
                 initializer_range=None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = hidden_size ** -0.5 if initializer_range is None else initializer_range

    def build(self, input_shape):
        """Build shared token embedding layer
        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        self.weight = self.add_weight(
            "weight", shape=[self.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range)
        )
        super().build(input_shape)

    def call(self, inputs, mode="embedding"):
        """Get token embeddings of inputs.
        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
                shape [batch_size, length, embedding_size]; (2) mode == "linear", output
                linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, input_ids):
        """Applies embedding based on inputs tensor."""
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [..., hidden_size]
            Returns:
                float32 tensor with shape [..., vocab_size].
        """
        first_dims = get_shape(inputs)[:-1]

        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])


class T5LayerNorm(Layer):
    def __init__(self,
                epsilon=1e-6, 
                **kwargs):
        """ Construct a layernorm module in the T5 style
            No bias and no substraction of mean.
        """
        super().__init__(**kwargs)
        self.variance_epsilon = epsilon

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.weight = self.add_weight("weight", shape=(input_shape[-1],), initializer="ones")
        super().build(input_shape)

    def call(self, x):
        variance = tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(variance + self.variance_epsilon)
        return self.weight * x


class DenseReLUDense(Layer):
    def __init__(self, 
                 d_ff,
                 d_model,
                 dropout_rate,
                 **kwargs):
        super().__init__(**kwargs)
        self.wi = Dense(d_ff, use_bias=False, name="wi")
        self.wo = Dense(d_model, use_bias=False, name="wo")
        self.dropout = Dropout(dropout_rate)
        self.act = ReLU()

    def call(self,
             hidden_states, 
             training=False):

        h = self.wi(hidden_states)
        h = self.act(h)
        h = self.dropout(h, training=training)
        h = self.wo(h)
        return h


class FeedForward(Layer):
    def __init__(self, 
                 d_ff,
                 d_model,
                 dropout_rate,
                 layer_norm_epsilon,
                 **kwargs):

        super().__init__(**kwargs)
        self.DenseReLUDense = DenseReLUDense(d_ff,
                                             d_model,
                                             dropout_rate, 
                                             name="dense_relu_dense")
        self.layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon, 
                                      name="layer_norm")
        self.dropout = Dropout(dropout_rate)

    def call(self,
             hidden_states, 
             training=False):

        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReLUDense(norm_x, training=training)
        layer_output = hidden_states + self.dropout(y, training=training)
        return layer_output

class MultiHeadAttention(Layer):
    def __init__(self,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 role_type,
                 num_relative_buckets,
                 max_relative_distance,
                 r3_vocab_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_relative_buckets = num_relative_buckets
        self.max_relative_distance = max_relative_distance
        self.d_model = d_model
        self.d_kv = d_kv
        self.n_heads = num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.enable_tp = role_type is not None
        self.role_type = role_type

        #query, key, and value mapping
        self.q = Dense(self.inner_dim, use_bias=False, name="q")
        self.k = Dense(self.inner_dim, use_bias=False, name="k")
        self.v = Dense(self.inner_dim, use_bias=False, name="v")
        
        #role mapping (if tp-enabled)
        if self.enable_tp:
            assert self.role_type in ['self_role', 'relative_role', 'attn_relative_role'], "Role type not supported!"
            
            if self.role_type == 'self_role':
                self.r = Dense(self.inner_dim, use_bias=False, name="r")
            
            elif self.role_type == 'relative_role':
                self.r = RelativeRole(d_kv=self.d_kv,
                                      n_heads=self.n_heads,
                                      vocab_size=r3_vocab_size,
                                      dropout_rate=dropout_rate, 
                                      name='r3')
            
            elif self.role_type == 'attn_relative_role':
                self.r = AttentiveRelativeRole(d_kv=self.d_kv,
                                               n_heads=self.n_heads,
                                               vocab_size=r3_vocab_size,
                                               dropout_rate=dropout_rate, 
                                               name='attentive_r3')

        self.o = Dense(self.d_model, use_bias=False, name="o")
        self.dropout = Dropout(dropout_rate)

        if self.num_relative_buckets is not None:
            self.relative_attention_bias = Embedding(self.num_relative_buckets, 
                                                     self.n_heads, 
                                                     name="relative_attention_bias")

    def _relative_position_bucket(self,
                                  relative_position, 
                                  bidirectional, 
                                  num_buckets, 
                                  max_distance):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += tf.dtypes.cast(tf.math.less(n, 0), tf.int32) * num_buckets
            n = tf.math.abs(n)
        else:
            n = tf.math.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(n, max_exact)
        val_if_large = max_exact + tf.dtypes.cast(
            tf.math.log(tf.dtypes.cast(n, tf.float32) / max_exact)
            / np.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            tf.int32,
        )
        val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
        ret += tf.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, 
                     qlen, 
                     klen,
                     bidirectional):
        """ Compute binned relative position bias """
        context_position = tf.range(qlen)[:, None]
        memory_position = tf.range(klen)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(relative_position=relative_position, 
                                                   bidirectional=bidirectional, 
                                                   num_buckets=self.num_relative_buckets,
                                                   max_distance=self.max_relative_distance)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)  # shape (1, num_heads, qlen, klen)
        return values

    def _split_heads(self, x, bs):
        """ split heads and rearrange elements """
        """ output shape: (bs, n_heads, seq_len, d_kv) """
        return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, self.d_kv)), perm=(0, 2, 1, 3))

    def _join_heads(self, x, bs):
        """ split heads and rearrange elements """
        """ output shape: (bs, seq_len, inner_dim) """
        return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.inner_dim))

    def self_attention(self,
                     inputs,
                     mask=None,
                     bidirectional=True,
                     position_bias=None,
                     position_embeddings=None,
                     past_key_value_state=None,
                     use_cache=False,
                     training=False):
        
        bs, qlen, dim = get_shape(inputs)
        if past_key_value_state is not None:
            error_message = "past_key_value_state should have 2 past states: keys and values. Got {} past states".format(len(past_key_value_state))
            assert (len(past_key_value_state) == 2), error_message
            seq_len = qlen + get_shape(past_key_value_state[0])[2]
        else:
            seq_len = qlen

        q = self.q(inputs)  # (bs, seq_len, inner_dim)
        k = self.k(inputs)  # (bs, seq_len, inner_dim)
        v = self.v(inputs)  # (bs, seq_len, inner_dim)

        q = self._split_heads(q, bs) # (bs, n_heads, seq_len, dim_per_head)
        k = self._split_heads(k, bs) # (bs, n_heads, seq_len, dim_per_head)
        v = self._split_heads(v, bs) # (bs, n_heads, seq_len, dim_per_head)

        loss = 0.
        vq_outs = {}
        if self.enable_tp:
            if self.role_type == 'self_role':
                r = self.r(inputs)  # (bs, seq_len, inner_dim)
                r = self._split_heads(r, bs) # (bs, n_heads, seq_len, dim_per_head)
            
            elif self.role_type == 'relative_role':
                r, vq_outs = self.r(query=inputs, 
                                 key=inputs, 
                                 q_position_embeddings=position_embeddings, 
                                 k_position_embeddings=position_embeddings,
                                 training=training)  # (bs, n_heads, seq_len, dim_per_head)
                loss = vq_outs['loss']
            elif self.role_type == 'attn_relative_role':
                r, vq_outs = self.r(query=inputs, 
                                 key=inputs, 
                                 value=inputs, 
                                 mask=mask,
                                 q_position_embeddings=position_embeddings,
                                 kv_position_embeddings=position_embeddings,
                                 training=training)  # (bs, n_heads, rlen, dim_per_head)
                loss = vq_outs['loss']
        if past_key_value_state is not None:
            k_, v_ = past_key_value_state
            k = tf.concat([k_, k], axis=2)  # (bs, n_heads, seq_len, dim_per_head)
            v = tf.concat([v_, v], axis=2)  # (bs, n_heads, seq_len, dim_per_head)

        if tf.is_tensor(use_cache):
            if hasattr(use_cache, "numpy"):
                use_cache = bool(use_cache.numpy())
            else:
                use_cache = True

        if use_cache:
            present_key_value_state = (k, v)
        else:
            present_key_value_state = None

        if position_bias is None:
            if self.num_relative_buckets is None:
                raise ValueError("No position_bias provided and no weights to compute position_bias")
            position_bias = self.compute_bias(qlen=seq_len, klen=seq_len, bidirectional=bidirectional)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value_state is not None:
                position_bias = position_bias[:, :, -1:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (bs, n_heads, seq_len, seq_len)


        scores = tf.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, seq_len, seq_len)
        scores += position_bias
        attention_weights = tf.nn.softmax(scores, axis=-1)  # (bs, n_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights, training=training)  # (bs, n_heads, seq_len, seq_len)

        hidden_states = tf.matmul(attention_weights, v)  # (bs, n_heads, seq_len, dim_per_head)
        
        if self.enable_tp:
            hidden_states = tf.multiply(hidden_states, r)  # (bs, n_heads, seq_len, dim_per_head)

        hidden_states = self._join_heads(hidden_states, bs)  # (bs, seq_len, dim)
        hidden_states = self.o(hidden_states)

        outputs = {
                   'hidden_states': hidden_states,
                   'key_value_state': present_key_value_state,
                   'attention_weights': attention_weights,
                   'position_bias': position_bias,
                   'vq_states': vq_outs,
                   'loss': loss,
                   }

        return outputs

    def cross_attention(self,
                         query,
                         key,
                         value,
                         mask=None,
                         bidirectional=False,
                         use_cache=False,
                         position_bias=None,
                         position_embeddings=None,
                         kv_position_embeddings=None,
                         query_length=None,
                         past_key_value_state=None,
                         training=False):
        
        bs, qlen, dim = get_shape(query)
        klen = get_shape(key)[1]

        if past_key_value_state is not None:
            error_message = "past_key_value_state should have 2 past states: keys and values. Got {} past states".format(len(past_key_value_state))
            assert (len(past_key_value_state) == 2), error_message
            real_qlen = qlen + get_shape(past_key_value_state[0])[2] if query_length is None else query_length
        else:
            real_qlen = qlen

        
        q = self.q(query)  # (bs, qlen, inner_dim)
        q = self._split_heads(q, bs) # (bs, n_heads, qlen, dim_per_head)

        loss = 0.
        vq_outs = {}
        if self.enable_tp:
            if self.role_type == 'self_role':
                r = self.r(query)  # (bs, rlen, inner_dim)
                r = self._split_heads(r, bs) # (bs, n_heads, rlen, dim_per_head)

            elif self.role_type == 'relative_role':
                r, vq_outs = self.r(query=query,
                                 key=key,
                                 q_position_embeddings=position_embeddings,
                                 k_position_embeddings=kv_position_embeddings,
                                 training=training)  # (bs, n_heads, rlen, dim_per_head)
                loss = vq_outs['loss']
            elif self.role_type == 'attn_relative_role':
                r, vq_outs = self.r(query=query, 
                                 key=key, 
                                 value=value, 
                                 mask=mask,
                                 q_position_embeddings=position_embeddings,
                                 kv_position_embeddings=kv_position_embeddings,
                                 training=training)  # (bs, n_heads, rlen, dim_per_head)
                loss = vq_outs['loss']
        if past_key_value_state is None:
            k = self.k(key)  # (bs, klen, inner_dim)
            v = self.v(value)  # (bs, klen, inner_dim)
            k = self._split_heads(k, bs) # (bs, n_heads, klen, dim_per_head)
            v = self._split_heads(v, bs) # (bs, n_heads, vlen, dim_per_head)
        else:
            k, v = past_key_value_state

        if tf.is_tensor(use_cache):
            if hasattr(use_cache, "numpy"):
                use_cache = bool(use_cache.numpy())
            else:
                use_cache = True

        if use_cache:
            present_key_value_state = (k, v)
        else:
            present_key_value_state = None

        if position_bias is None:
            if self.num_relative_buckets is None:
                raise ValueError("No position_bias provided and no weights to compute position_bias")
            position_bias = self.compute_bias(qlen=qlen, klen=klen, bidirectional=bidirectional)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value_state is not None:
                position_bias = position_bias[:, :, -1:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (bs, n_heads, seq_len, seq_len)


        scores = tf.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, seq_len, seq_len)
        scores += position_bias
        attention_weights = tf.nn.softmax(scores, axis=-1)  # (bs, n_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights, training=training)  # (bs, n_heads, seq_len, seq_len)

        hidden_states = tf.matmul(attention_weights, v)  # (bs, n_heads, seq_len, dim_per_head)

        if self.enable_tp:
            hidden_states = tf.multiply(hidden_states, r)  # (bs, n_heads, qlen, dim_per_head)

        hidden_states = self._join_heads(hidden_states, bs)  # (bs, seq_len, dim)
        hidden_states = self.o(hidden_states)

        outputs = {
                   'hidden_states': hidden_states,
                   'key_value_state': present_key_value_state,
                   'attention_weights': attention_weights,
                   'position_bias': position_bias,
                   'vq_states': vq_outs,
                   'loss': loss,
                   }

        return outputs

class AttentionLayer(Layer):
    def __init__(self,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 layer_norm_epsilon,
                 role_type,
                 attention_type,
                 num_relative_buckets,
                 max_relative_distance,
                 r3_vocab_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        assert attention_type in {'self_attention', 'cross_attention'}, "Attention type not supported!"
        
        self.is_self_attention = attention_type == 'self_attention'
        self.attention = MultiHeadAttention(d_model=d_model,
                                            d_kv=d_kv,
                                            num_heads=num_heads,
                                            dropout_rate=dropout_rate,
                                            role_type=role_type,
                                            num_relative_buckets=num_relative_buckets,
                                            max_relative_distance=max_relative_distance,
                                            r3_vocab_size=r3_vocab_size,
                                            name="multi_head_attention")
        self.layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon, 
                                      name="layer_norm")
        self.dropout = Dropout(dropout_rate)

    def self_attention(self,
                       inputs,
                       bidirectional=True,
                       attention_mask=None,
                       position_bias=None,
                       position_embeddings=None,
                       past_key_value_state=None,
                       use_cache=False,
                       training=False):
        
        norm_x = self.layer_norm(inputs)
        attention_outputs = self.attention.self_attention(inputs=norm_x,
                                                          bidirectional=bidirectional,
                                                          mask=attention_mask,
                                                          position_bias=position_bias,
                                                          position_embeddings=position_embeddings,
                                                          past_key_value_state=past_key_value_state,
                                                          use_cache=use_cache,
                                                          training=training)
        hidden_states = attention_outputs['hidden_states']
        hidden_states = inputs + self.dropout(hidden_states, training=training)

        #update hidden states
        attention_outputs['hidden_states'] = hidden_states

        return attention_outputs


    def cross_attention(self,
                        query,
                        key,
                        value,
                        bidirectional=False,
                        attention_mask=None,
                        position_bias=None,
                        position_embeddings=None,
                        kv_position_embeddings=None,
                        past_key_value_state=None,
                        query_length=None,
                        use_cache=False,
                        training=False):
        
        norm_query = self.layer_norm(query)
        attention_outputs = self.attention.cross_attention(query=norm_query,
                                                           key=key,
                                                           value=value,
                                                           bidirectional=bidirectional,
                                                           mask=attention_mask,
                                                           position_bias=position_bias,
                                                           position_embeddings=position_embeddings,
                                                           kv_position_embeddings=kv_position_embeddings,
                                                           past_key_value_state=past_key_value_state,
                                                           query_length=query_length,
                                                           use_cache=use_cache,
                                                           training=training)

        hidden_states = attention_outputs['hidden_states']
        hidden_states = query + self.dropout(hidden_states, training=training)

        #update hidden states
        attention_outputs['hidden_states'] = hidden_states
        
        return attention_outputs

    def call(self,
             inputs,
             key=None,
             value=None,
             bidirectional=False,
             attention_mask=None,
             position_bias=None,
             position_embeddings=None,
             kv_position_embeddings=None,
             past_key_value_state=None,
             query_length=None,
             use_cache=False,
             training=False):

        if self.is_self_attention:
            return self.self_attention(inputs=inputs,
                                       bidirectional=bidirectional,
                                       attention_mask=attention_mask,
                                       position_bias=position_bias,
                                       position_embeddings=position_embeddings,
                                       past_key_value_state=past_key_value_state,
                                       use_cache=use_cache,
                                       training=training) 
        else:
            return self.cross_attention(query=inputs,
                                        key=key,
                                        value=value,
                                        bidirectional=bidirectional,
                                        attention_mask=attention_mask,
                                        position_bias=position_bias,
                                        position_embeddings=position_embeddings,
                                        kv_position_embeddings=kv_position_embeddings,
                                        past_key_value_state=past_key_value_state,
                                        query_length=query_length,
                                        use_cache=use_cache,
                                        training=training)


class EncoderLayer(Layer):
    def __init__(self,
                 d_model,
                 d_kv,
                 d_ff,
                 num_heads,
                 dropout_rate,
                 layer_norm_epsilon,
                 role_type,
                 num_relative_buckets,
                 max_relative_distance,
                 r3_vocab_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.self_attention = AttentionLayer(d_model=d_model,
                                             d_kv=d_kv,
                                             num_heads=num_heads,
                                             dropout_rate=dropout_rate,
                                             layer_norm_epsilon=layer_norm_epsilon,
                                             num_relative_buckets=num_relative_buckets,
                                             max_relative_distance=max_relative_distance,
                                             r3_vocab_size=r3_vocab_size,
                                             role_type=role_type,
                                             attention_type='self_attention',
                                             name='self_attention')
        self.feed_forward = FeedForward(d_ff=d_ff,
                                        d_model=d_model,
                                        dropout_rate=dropout_rate,
                                        layer_norm_epsilon=layer_norm_epsilon,
                                        name='feed_forward')

    def call(self,
             inputs,
             attention_mask=None,
             position_bias=None,
             position_embeddings=None,
             training=False):

        attention_outputs = self.self_attention(inputs=inputs,
                                                bidirectional=True,
                                                attention_mask=attention_mask,
                                                position_bias=position_bias,
                                                position_embeddings=position_embeddings,
                                                training=training)

        hidden_states = attention_outputs['hidden_states']

        # Apply Feed Forward layer
        hidden_states = self.feed_forward(hidden_states, training=training)

        outputs = {
                   'hidden_states': hidden_states,
                   'self_attention_weights': attention_outputs['attention_weights'],
                   'self_position_bias': attention_outputs['position_bias'],
                   'self_attention_vq_states': attention_outputs['vq_states'],
                   'loss': attention_outputs['loss'],
                   }

        return outputs

class DecoderLayer(Layer):
    def __init__(self,
                 d_model,
                 d_kv,
                 d_ff,
                 num_heads,
                 dropout_rate,
                 layer_norm_epsilon,
                 self_attn_role_type,
                 cross_attn_role_type,
                 num_self_relative_buckets,
                 max_self_relative_distance,
                 num_cross_relative_buckets,
                 max_cross_relative_distance,
                 self_attn_r3_vocab_size=None,
                 cross_attn_r3_vocab_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.self_attention = AttentionLayer(d_model=d_model,
                                             d_kv=d_kv,
                                             num_heads=num_heads,
                                             dropout_rate=dropout_rate,
                                             layer_norm_epsilon=layer_norm_epsilon,
                                             num_relative_buckets=num_self_relative_buckets,
                                             max_relative_distance=max_self_relative_distance,
                                             r3_vocab_size=self_attn_r3_vocab_size,
                                             role_type=self_attn_role_type,
                                             attention_type='self_attention',
                                             name='self_attention')
        self.cross_attention = AttentionLayer(d_model=d_model,
                                              d_kv=d_kv,
                                              num_heads=num_heads,
                                              dropout_rate=dropout_rate,
                                              layer_norm_epsilon=layer_norm_epsilon,
                                              num_relative_buckets=num_cross_relative_buckets,
                                              max_relative_distance=max_cross_relative_distance,
                                              r3_vocab_size=cross_attn_r3_vocab_size,
                                              role_type=cross_attn_role_type,
                                              attention_type='cross_attention',
                                              name='cross_attention')
        self.feed_forward = FeedForward(d_ff=d_ff,
                                        d_model=d_model,
                                        dropout_rate=dropout_rate,
                                        layer_norm_epsilon=layer_norm_epsilon,
                                        name='feed_forward')

    def call(self,
             inputs,
             encoder_hidden_states,
             encoder_position_embeddings=None,
             attention_mask=None,
             encoder_attention_mask=None,
             position_bias=None,
             position_embeddings=None,
             encoder_decoder_position_bias=None,
             past_key_value_state=None,
             use_cache=False,
             training=False):
        
        if past_key_value_state is not None:
            expected_num_past_key_value_states = 2 if encoder_hidden_states is None else 4
            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_value_states,
                "2 (past / key) for cross attention" if expected_num_past_key_value_states == 4 else "",
                len(past_key_value_state))
            assert len(past_key_value_state) == expected_num_past_key_value_states, error_message

            self_attn_past_key_value_state = past_key_value_state[:2]
            cross_attn_past_key_value_state = past_key_value_state[2:]
        else:
            self_attn_past_key_value_state, cross_attn_past_key_value_state = None, None

        self_attention_outputs = self.self_attention(inputs=inputs,
                                                     bidirectional=False,
                                                     attention_mask=attention_mask,
                                                     position_bias=position_bias,
                                                     position_embeddings=position_embeddings,
                                                     past_key_value_state=self_attn_past_key_value_state,
                                                     use_cache=use_cache,
                                                     training=training)
        
        decoder_hidden_states = self_attention_outputs['hidden_states']
        self_attn_key_value_state = self_attention_outputs['key_value_state'] # None or (k_self,v_self)

        # the actual query length is unknown for cross attention
        # if using past key value states. Need to inject it here
        if self_attn_key_value_state is not None:
            query_length = get_shape(self_attn_key_value_state[0])[2]
        else:
            query_length = None

        cross_attention_outputs = self.cross_attention(inputs=decoder_hidden_states,
                                                       key=encoder_hidden_states,
                                                       value=encoder_hidden_states,
                                                       bidirectional=False,
                                                       attention_mask=encoder_attention_mask,
                                                       position_bias=encoder_decoder_position_bias,
                                                       position_embeddings=position_embeddings,
                                                       kv_position_embeddings=encoder_position_embeddings,
                                                       past_key_value_state=cross_attn_past_key_value_state,
                                                       query_length=query_length,
                                                       use_cache=use_cache,
                                                       training=training)
        
        decoder_hidden_states = cross_attention_outputs['hidden_states']
        cross_attn_key_value_state = cross_attention_outputs['key_value_state']

        # Combine self attn and cross attn key value states
        if self_attn_key_value_state is not None:
            present_key_value_state = self_attn_key_value_state + cross_attn_key_value_state # (k_self,v_self,k_cross,v_cross)
        else:
            present_key_value_state = None

        # Apply Feed Forward layer
        decoder_hidden_states = self.feed_forward(decoder_hidden_states, training=training)
        
        outputs = {
                   'hidden_states': decoder_hidden_states,
                   'key_value_state': present_key_value_state,
                   'self_attention_weights': self_attention_outputs['attention_weights'],
                   'self_position_bias': self_attention_outputs['position_bias'],
                   'self_attention_vq_states': self_attention_outputs['vq_states'],
                   'cross_attention_weights': cross_attention_outputs['attention_weights'],
                   'cross_position_bias': cross_attention_outputs['position_bias'],
                   'cross_attention_vq_states': cross_attention_outputs['vq_states'],
                   'loss': self_attention_outputs['loss'] + cross_attention_outputs['loss'],
                   }
        
        return outputs

class T5Encoder(Layer):
    def __init__(self, 
                 d_model,
                 d_kv,
                 d_ff,
                 num_layers,
                 num_heads,
                 role_types,
                 dropout_rate,
                 layer_norm_epsilon,
                 num_relative_buckets,
                 max_relative_distance,
                 max_tmp_positions,
                 max_spt_positions,
                 r3_vocab_size=None,
                 use_embeddings=False,
                 vocab_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        
        if use_embeddings:
            self.embeddings = Embeddings(vocab_size=vocab_size,
                                         hidden_size=d_model,
                                         name='encoder_embeddings')
        else:
            self.embeddings = None
        
        self.d_model = d_model
        self.num_hidden_layers = num_layers
        self.relative_buckets = [num_relative_buckets]+[None]*(num_layers-1)
        self.relative_distances = [max_relative_distance]+[None]*(num_layers-1)
        self.r3_vocab_sizes = [r3_vocab_size]*num_layers
        self.use_pos_embds = not all([(r is None) or (r == 'self_role') for r in role_types])

        if self.use_pos_embds:
            self.spatio_temporal_embeddings = SpatioTemporalEmbeddings(hidden_size=d_model,
                                                                       max_temporal_positions=max_tmp_positions,
                                                                       max_spatial_centers=max_spt_positions,
                                                                       name='spatio_temporal_embeddings')

        self.layers = [
                      EncoderLayer(d_model=d_model,
                                   d_kv=d_kv,
                                   d_ff=d_ff,
                                   num_heads=num_heads,
                                   dropout_rate=dropout_rate,
                                   layer_norm_epsilon=layer_norm_epsilon,
                                   role_type=role_types[n],
                                   num_relative_buckets=self.relative_buckets[n], 
                                   max_relative_distance=self.relative_distances[n],
                                   r3_vocab_size=self.r3_vocab_sizes[n],
                                   name="layer_{}".format(n))
                      for n in range(self.num_hidden_layers)
                      ]

        self.final_layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon, name="final_layer_norm")
        self.dropout = Dropout(dropout_rate)

    def call(
        self,
        inputs,
        input_pos_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        training=False):

        if inputs is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both inputs and inputs_embeds at the same time")
        elif inputs is not None:
            input_shape = get_shape(inputs)
            inputs = tf.reshape(inputs, (-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = get_shape(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either inputs or inputs_embeds")

        if inputs_embeds is None:
            assert self.embeddings is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embeddings(inputs)

        batch_size, seq_length = input_shape
        mask_seq_length = seq_length
        if attention_mask is None:
            attention_mask = tf.fill((batch_size, mask_seq_length), 1)

        # Provided a padding mask of dimensions [batch_size, mask_seq_length]
        # make the mask broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -1e9

        all_hidden_states = ()
        all_attentions = ()
        all_vq_states = ()
        all_losses = 0.
        position_bias = None
        position_embeddings = None

        hidden_states = self.dropout(inputs_embeds, training=training)
        if self.use_pos_embds:
            position_embeddings = self.spatio_temporal_embeddings(input_pos_ids, training)

        for n, layer in enumerate(self.layers):
            #temporary --- to reduce memory consumption
            #all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(inputs=hidden_states,
                                  attention_mask=attention_mask,
                                  position_bias=position_bias,
                                  position_embeddings=position_embeddings,
                                  training=training)
           
            # layer_outputs is a dictionary with the following keys:
            # hidden_states, key_value_state, self_attention_weights, self_position_bias

            hidden_states = layer_outputs['hidden_states']
            if n==0:
                position_bias = layer_outputs['self_position_bias']

            all_attentions = all_attentions + (layer_outputs['self_attention_weights'],)
            all_vq_states = all_vq_states + (layer_outputs['self_attention_vq_states'],)
            all_losses = all_losses + layer_outputs['loss']

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        # Add last layer
        all_hidden_states = all_hidden_states + (hidden_states,)

        #calculate layer loss (if any)
        loss = all_losses

        outputs = {
                   'hidden_states': all_hidden_states,
                   'attention_weights': all_attentions,
                   'position_embeddings': position_embeddings,
                   'attention_vq_states': all_vq_states,
                   'loss': loss,
                   }
        
        return outputs


class T5Decoder(Layer):
    def __init__(self, 
                 d_model,
                 d_kv,
                 d_ff,
                 num_layers,
                 num_heads,
                 self_role_types,
                 cross_role_types,
                 num_self_relative_buckets,
                 max_self_relative_distance,
                 num_cross_relative_buckets,
                 max_cross_relative_distance,
                 max_tmp_positions,
                 dropout_rate,
                 layer_norm_epsilon,
                 vocab_size,
                 self_r3_vocab_size=None,
                 cross_r3_vocab_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.embeddings = Embeddings(vocab_size=vocab_size,
                                     hidden_size=d_model,
                                     name='decoder_embeddings')

        self.d_model = d_model
        self.num_hidden_layers = num_layers
        
        self.self_relative_buckets = [num_self_relative_buckets]+[None]*(num_layers-1)
        self.self_relative_distances = [max_self_relative_distance]+[None]*(num_layers-1)
        self.cross_relative_buckets = [num_cross_relative_buckets]+[None]*(num_layers-1)
        self.cross_relative_distances = [max_cross_relative_distance]+[None]*(num_layers-1)
        self.self_r3_vocab_sizes = [self_r3_vocab_size] * num_layers
        self.cross_r3_vocab_sizes = [cross_r3_vocab_size] * num_layers

        self.use_pos_embds = not (all([(r is None) or (r == 'self_role') for r in self_role_types])
                             and all([(r is None) or (r == 'self_role') for r in cross_role_types]))

        if self.use_pos_embds:
            self.temporal_embeddings = TemporalEmbeddings(hidden_size=d_model,
                                                          max_temporal_positions=max_tmp_positions,
                                                          name='temporal_embeddings')

        self.layers = [
                      DecoderLayer(d_model=d_model,
                                   d_kv=d_kv,
                                   d_ff=d_ff,
                                   num_heads=num_heads,
                                   dropout_rate=dropout_rate,
                                   layer_norm_epsilon=layer_norm_epsilon,
                                   self_attn_role_type=self_role_types[n],
                                   cross_attn_role_type=cross_role_types[n],
                                   num_self_relative_buckets=self.self_relative_buckets[n],
                                   max_self_relative_distance=self.self_relative_distances[n],
                                   num_cross_relative_buckets=self.cross_relative_buckets[n],
                                   max_cross_relative_distance=self.cross_relative_distances[n],
                                   self_attn_r3_vocab_size=self.self_r3_vocab_sizes[n],
                                   cross_attn_r3_vocab_size=self.cross_r3_vocab_sizes[n],
                                   name="layer_{}".format(n))
                      for n in range(self.num_hidden_layers)
                      ]

        self.final_layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon, name="final_layer_norm")
        self.dropout = Dropout(dropout_rate)


    def call(self,
             inputs,
             encoder_hidden_states,
             encoder_position_embeddings=None,
             attention_mask=None,
             encoder_attention_mask=None,
             inputs_embeds=None,
             past_key_value_states=None,
             use_cache=False,
             training=False):

        if inputs is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both inputs and inputs_embeds at the same time")
        elif inputs is not None:
            input_shape = get_shape(inputs)
            inputs = tf.reshape(inputs, (-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = get_shape(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either inputs or inputs_embeds")

        batch_size, seq_length = input_shape

        if inputs_embeds is None:
            assert self.embeddings is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embeddings(inputs)

        if past_key_value_states is not None:
            assert seq_length == 1, "Input shape is {}, but should be {} when using past_key_value_sates".format(
                input_shape, (batch_size, 1)
            )
            # required mask seq length can be calculated via length of past
            # key value states and seq_length = 1 for the last token
            mask_seq_length = get_shape(past_key_value_states[0][0])[2] + seq_length
        else:
            mask_seq_length = seq_length

        if attention_mask is None:
            attention_mask = tf.fill((batch_size, mask_seq_length), 1)

        if encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = get_shape(encoder_hidden_states)[1]
            encoder_attention_mask = tf.fill((batch_size, encoder_seq_length), 1)

        # initialize past_key_value_states with `None` if past does not exist
        if past_key_value_states is None:
            past_key_value_states = [None] * self.num_hidden_layers

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        num_dims_attention_mask = len(get_shape(attention_mask))
        if num_dims_attention_mask == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif num_dims_attention_mask == 2:
            # Provided a padding mask of dimensions [batch_size, mask_seq_length]
            # - in a decoder, apply a causal mask in addition to the padding mask
            seq_ids = tf.range(mask_seq_length)
            causal_mask = tf.less_equal(
                tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)), seq_ids[None, :, None],
            )
            causal_mask = tf.cast(causal_mask, dtype=tf.float32)
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            if past_key_value_states[0] is not None:
                extended_attention_mask = extended_attention_mask[:, :, -1:, :]

        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        if encoder_attention_mask is not None:
            # If a 2D or 3D attention mask is provided for the cross-attention
            # we need to make broadcastabe to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=tf.float32)
            num_dims_encoder_attention_mask = len(get_shape(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            encoder_extended_attention_mask = None

        present_key_value_states = ()
        all_hidden_states = ()
        all_self_attentions = ()
        all_self_attention_vq_states = ()
        all_cross_attentions = ()
        all_cross_attention_vq_states = ()
        all_losses = 0.
        position_bias = None
        encoder_decoder_position_bias = None
        position_embeddings = None

        hidden_states = self.dropout(inputs_embeds, training=training)
        if self.use_pos_embds:
            position_embeddings = self.temporal_embeddings(inputs_embeds, training)

        for n, (layer, past_key_value_state) in enumerate(zip(self.layers, past_key_value_states)):
            #temporary --- to reduce memory consumption
            #all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer(inputs=hidden_states,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_position_embeddings=encoder_position_embeddings,
                                  attention_mask=extended_attention_mask,
                                  encoder_attention_mask=encoder_extended_attention_mask,
                                  position_bias=position_bias,
                                  position_embeddings=position_embeddings,
                                  encoder_decoder_position_bias=encoder_decoder_position_bias,
                                  past_key_value_state=past_key_value_state,
                                  use_cache=use_cache,
                                  training=training)

            # layer_outputs is a dictionary with the following keys:
            # hidden_states, key_value_state, self_attention_weights, self_position_bias, cross_attention_weights, cross_position_bias
            hidden_states = layer_outputs['hidden_states']
            present_key_value_state = layer_outputs['key_value_state']

            if n == 0:
                # We share the position biases between the layers - the first layer store them
                position_bias = layer_outputs['self_position_bias']
                encoder_decoder_position_bias = layer_outputs['cross_position_bias']

            # append next layer key value states
            present_key_value_states = present_key_value_states + (present_key_value_state,)

            all_self_attentions = all_self_attentions + (layer_outputs['self_attention_weights'],)
            all_cross_attentions = all_cross_attentions + (layer_outputs['cross_attention_weights'],)
            all_self_attention_vq_states = all_self_attention_vq_states + (layer_outputs['self_attention_vq_states'],)
            all_cross_attention_vq_states = all_cross_attention_vq_states + (layer_outputs['cross_attention_vq_states'],)
            all_losses = all_losses + layer_outputs['loss']

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        # Add last layer
        all_hidden_states = all_hidden_states + (hidden_states,)

        #calculate layer loss (if any)
        #loss = tf.reduce_mean(all_losses)
        loss = all_losses

        outputs = {
                   'hidden_states': all_hidden_states,
                   'key_value_states': present_key_value_states,
                   'self_attention_weights': all_self_attentions,
                   'cross_attention_weights': all_cross_attentions,
                   'self_attention_vq_states': all_self_attention_vq_states,
                   'cross_attention_vq_states': all_cross_attention_vq_states,
                   'loss': loss,
                   }
        
        return outputs

