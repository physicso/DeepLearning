import tensorflow as tf
from keras import layers, models

# Define important parameters.
n_hidden = 64
n_classes = 2
n_layers = 4
batch_size = 512
max_length = 50
frame_size = 300


# Obtain the true (non-zero) length of the time series data.
def length(seq):
    used = tf.sign(tf.reduce_max(tf.abs(seq), axis=2))
    leng = tf.reduce_sum(used, axis=1)
    leng = tf.cast(leng, tf.int32)
    return leng


# Obtain the characterization vector corresponding to the actual length.
def last_relevant(output, length):
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, n_hidden])
    relevant = tf.gather(flat, index)
    return relevant


def rnn(sequence):
    seq_length = length(sequence)
    layer = sequence
    for _ in range(n_layers):
        layer = layers.LSTM(n_hidden, return_sequences=True)(layer)
    last = last_relevant(layer, seq_length)
    pred = layers.Dense(2, activation='softmax')(last)
    return pred


sequence = layers.Input(shape=[max_length, frame_size], dtype=tf.float32)
label = layers.Input(shape=[n_classes, ], dtype=tf.float32)
prediction = rnn(sequence)
model = models.Model(sequence, prediction)
print(model.summary())
