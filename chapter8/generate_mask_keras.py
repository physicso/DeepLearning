import numpy as np
import tensorflow as tf


def generate_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, np.newaxis, np.newaxis, :]


def generate_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


input_tensor = [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
print(generate_padding_mask(input_tensor))
mask = generate_look_ahead_mask(3)
print(mask)
