import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
References:

    1. https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

    2. https://github.com/ukairia777/tensorflow-transformer/blob/main/Transformer_Korean_Chatbot.ipynb
"""

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        sines = tf.math.sin(angle_rads[:, 0::2])

        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pos_encoding': self.pos_encoding,
        })
        return config

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def ScaledDotProductAttention(query, key, value, mask):
    # query : (batch_size, num_heads, q_len, d_model/num_heads)
    # key   : (batch_size, num_heads, k_len, d_model/num_heads)
    # value : (batch_size, num_heads, v_len, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, sec_len)

    # Scaled dot product of Q and K
    tmp = tf.matmul(query, key, transpose_b=True)
    d_model = tf.cast( value.shape[-1], tf.float32 )
    # Scaling 
    logits = tmp / tf.math.sqrt(d_model)

    # Masking is required for decoder in order to 
    # filter out the the furure values of ouput while calculating the attention score
    if mask is not None:
        mask = np.zeros_like(logits)
        for i in range(mask.shape[0]):
            mask[i,i+1:] = -np.inf
        mask = tf.convert_to_tensor(mask)
        logits += mask
    
    # asttention score
    attention_score = tf.nn.softmax(logits, axis=-1)

    # output : (batch_size, num_heads, sec_len, d_model/num_heads)
    output = tf.matmul(attention_score, value)

    return output, attention_score


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        # depth is d_model divided by num_heads.
        self.depth = d_model // self.num_heads

        # WQ, WK, WV
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense   = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # WO
        self.dense = tf.keras.layers.Dense(units=d_model)

    # split q, k, v into num_heads pieces
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config

    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        if len(inputs) == 3:
            mask = None
        else:
            mask = inputs['mask']

        batch_size = tf.shape(query)[0]

        # 1. Passing WQ, WK, WV
        # q : (batch_size, q_len, d_model)
        # k : (batch_size, k_len, d_model)
        # v : (batch_size, v_len, d_model)

        query   = self.query_dense(query)
        key     = self.key_dense(key)
        value   = self.value_dense(value)

        # 2. split into heads
        # q : (batch_size, num_heads, q_len, d_model/num_heads)
        # k : (batch_size, num_heads, k_len, d_model/num_heads)
        # v : (batch_size, num_heads, v_len, d_model/num_heads)
        query   = self.split_heads(query, batch_size)
        key     = self.split_heads(key, batch_size)
        value   = self.split_heads(value, batch_size)

        # 3. scaled dot product attention using the "class ScaledDotProductAttention"
        scaled_attention, attention_score = ScaledDotProductAttention(query, key, value, mask)


        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, q_len, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # 5. WO dense layer
        # (batch_size, q_len, d_model)
        outputs = self.dense(concat_attention)

        return outputs, attention_score


def EncoderLayer(dff, d_model, num_heads, dropout, name="encoder_layer"):

    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    # multi-head attention ( first sub-layer, self-attention )
    # self-attention here means that query, key, value are all same as inputs.

    attention, attention_score = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': inputs, 'key': inputs, 'value':inputs}) # Q = K = V

    # Dropout
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)

    # Residual Connection
    attention = inputs + attention

    # Layer Normalization
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention)

    # Positionwise Feed Forward Layer
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # Dropout
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)

    # Residual Connection
    outputs = attention + outputs

    # Layer Normalization
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)( outputs)

    return tf.keras.Model(inputs=[inputs], outputs=[outputs, attention_score], name=name)



def encoder(sec_len, num_layers, dff, d_model, num_heads, dropout, name="encoder"):

    # sec_len    : sequence lenth
    # num_layers : Number of encoder layer
    # dff        : Size of the hidden layer in Positionwise Feed Forward layer
    # d_model    : Size of the embedding feature of the model, this one should be dividable by num_heads
    # num_heads  : number of heads for the multihead attention
    # dropout    : dropout rate between 0 to 1

    inputs = tf.keras.Input(shape=(10, d_model), name="inputs")

    embeddings = PositionalEncoding(sec_len, d_model)(inputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # stack the encoding layer
    for i in range(num_layers):
        outputs, attention_score = EncoderLayer(dff=dff, d_model=d_model, 
                     num_heads=num_heads, 
                     dropout=dropout, 
                     name="encoder_layer_{}".format(i))([outputs])
    return tf.keras.Model( inputs=inputs, outputs=[outputs,attention_score], name=name)



def transformer(input_size, num_layers, dff, d_model, num_heads, dropout, name="transformer"):

     # Input Layer
    sec_len, no_features = input_size
    inputs = tf.keras.Input(shape=(sec_len, no_features), name="inputs")

    # The embedding Layer is a simple dense layer since the input is a vector. 
    # Thus, no conversion is needed.
    embedded_output = tf.keras.layers.Dense(units= d_model, name="embedding")(inputs)

    
    enc_outputs, _= encoder(sec_len=sec_len, num_layers=num_layers, dff=dff,
        d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=embedded_output)

    # Dense layer to combine the output of the encoder unit
    outputs = tf.keras.layers.Dense(units=1, name="output")(enc_outputs)

    # inputs -> embedded_output -> encoder -> enc_outputs 
    outputs = outputs[:,:,-1]


    return tf.keras.Model(inputs=[inputs], outputs=outputs, name=name)




