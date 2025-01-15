import keras

# Sigmoid
Sigmoid = keras.activations.sigmoid(x)
# Tanh
tanh = keras.activations.tanh(x)
# ReLu
relu = keras.activations.relu(x)
# LeakyReLU
lkrelu = keras.layers.LeakyReLU(alpha=0.3)  # Here is the layer, where the parameter alpha represents the gradient when the input is less than 0.
# PReLU
prelu = keras.layers.PReLU(alpha_initializer='zeros')  # Note that here is the layer, where the parameter alpha_initializer indicates the initial value of the gradient when the input is less than 0.
