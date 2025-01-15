# Specify which GPUs to use via CUDA_VISIBLE_DEVICES, if specified, all GPUs are used by default.
# Note that the environment variables must be set before importing tensorflow and keras.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import tensorflow as tf
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import layers, models

# Reading MNIST datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Modeling neural networks
def fcnn(image_batch):
    h_fc1 = layers.Dense(200, input_dim=784)(image_batch)
    h_fc2 = layers.Dense(200)(h_fc1)
    _y = layers.Dense(10, activation='softmax')(h_fc2)
    return _y


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Creating distributed training strategies.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    # The model structure needs to be in scope.
    x = layers.Input(shape=(784,))
    y_ = layers.Input(shape=(10,))
    y = fcnn(x)
    model = models.Model(x, y)
    print(model.summary())
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))
