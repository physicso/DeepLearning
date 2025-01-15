import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, layers, models, optimizers

img_rows = 28
img_cols = 28
channels = 1

# MNIST image size.
img_shape = (img_rows, img_cols, channels)

# Noise vector size of generator.
z_dim = 100


# Used to draw an image.
def sample_images(generator, image_grid_rows=4, image_grid_columns=4):
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
    gen_imgs = generator.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1


def build_generator(z_dim):
    model = models.Sequential()
    # In order to input the vector into the CNN, a shape transformation is required.
    model.add(layers.Dense(256 * 7 * 7, input_dim=z_dim))
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
    # Notice the activation function here.
    model.add(layers.Activation('tanh'))
    return model


def build_discriminator(img_shape=(img_rows, img_cols, channels)):
    model = models.Sequential()
    model.add(
        layers.Conv2D(32,
                      kernel_size=3,
                      strides=2,
                      input_shape=img_shape,
                      padding='same'))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(
        layers.Conv2D(64,
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(
        layers.Conv2D(128,
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Flatten())
    # This is binary
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    model = models.Sequential()
    # Connect the generator to the discriminator.
    model.add(generator)
    model.add(discriminator)
    return model


# Build discriminators and generators.
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])
generator = build_generator(z_dim)
# When the generator is trained, the discriminator's parameters are locked.
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam())

losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):
    (X_train, _), (_, _) = datasets.mnist.load_data()
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)
    # All real data is labeled 1.
    real = np.ones((batch_size, 1))
    # All generated data is labeled 0.
    fake = np.zeros((batch_size, 1))
    for iteration in range(iterations):
        #  Training discriminator.
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)
        #  Training generator.
        z = np.random.normal(0, 1, (batch_size, z_dim))
        g_loss = gan.train_on_batch(z, real)
        if (iteration + 1) % sample_interval == 0:
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)
            # print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration + 1, d_loss, 100.0 * accuracy, g_loss))
            sample_images(generator)


train(10000, 128, 1000)
losses = np.array(losses)

# Draw the cost function of the generator and discriminator.
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.xticks(iteration_checkpoints, rotation=90)
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()
