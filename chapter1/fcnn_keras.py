import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import layers, models




# Used for visualizing MNIST image data.
def imshow(img):
    plt.imshow(np.reshape(img, [28, 28]))
    plt.show()


# Read the MNIST dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualize the first 5 images from the MNIST dataset.
for index in range(5):
    print(x_train[index].shape)
    print(y_train[index])
imshow(x_train[index])


# Build a neural network model.
def fcnn(image_batch):
    h_fc1 = layers.Dense(200, input_dim=784)(image_batch)
    h_fc2 = layers.Dense(200)(h_fc1)
    _y = layers.Dense(10, activation='softmax')(h_fc2)
    return _y


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.  # Normalize the training data.
x_test /= 255.  # Normalize the test data.
y_train = to_categorical(y_train, 10)  # Convert the training data labels to one-hot encoding.
y_test = to_categorical(y_test, 10)  # Convert the test data labels to one-hot encoding.

# In Keras, use the Input layer for data input.
x = layers.Input(shape=(784,))
y_ = layers.Input(shape=(10,))
y = fcnn(x)
model = models.Model(x, y)
print(model.summary())

# Use cross-entropy as the cost function, stochastic gradient descent as the optimization method, and accuracy as the monitoring metric.
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model for 5 epochs using mini-batches of size 64, and use the aforementioned test data as the validation set.
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))
