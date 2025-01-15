import numpy as np
import keras
import matplotlib.pyplot as plt

train_X = np.asarray([30.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0])
train_Y = np.asarray([320.0, 360.0, 400.0, 455.0, 490.0, 546.0, 580.0])
train_X /= 100.0  # Normalize the data.
train_Y /= 100.0


# Used for visualizing data points.
def plot_points(x, y, title_name):
    plt.title(title_name)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y)
    plt.show()


# Used for visualizing the linear fitting model.
def plot_line(W, b, title_name):
    plt.title(title_name)
    plt.xlabel('x')
    plt.ylabel('y')
    x = np.linspace(0.0, 2.0, num=100)
    y = W * x + b
    plt.plot(x, y)
    plt.show()


plot_points(train_X, train_Y, title_name='Training Points')

# Build a linear fitting model consisting of two parameters: slope and bias, which is equivalent to a fully connected layer.
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=1))

# The cost function uses mean squared error, and the optimization method employs stochastic gradient descent.
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model for 10 epochs, optimizing with single samples.
history = model.fit(x=train_X, y=train_Y, batch_size=1, epochs=10)

plot_line(model.get_weights()[0][0][0], model.get_weights()[1][0], title_name='Current Model')
