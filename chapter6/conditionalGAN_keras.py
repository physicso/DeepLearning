import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, layers, models, optimizers

img_rows = 28
img_cols = 28
channels = 1
num_classes = 10

# MNIST image size.
img_shape = (img_rows, img_cols, channels)

# Noise vector size of generator.
z_dim = 100
 

# Used to draw generated images, category information is added here.
def sample_images(image_grid_rows=2, image_grid_columns=5):
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
    labels = np.arange(0, 10).reshape(-1, 1)
    gen_imgs = generator.predict([z, labels])
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(10, 4),
                            sharey=True,
                            sharex=True)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title("Digit: %d" % labels[cnt])
            cnt += 1


def build_generator(z_dim):
    model = models.Sequential()
    # In order to input the vector into the CNN, a shape transformation is required
    model.add(layers.Dense(256 * 7 * 7, input_dim=z_dim))
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
    # Notice the activation function here
    model.add(layers.Activation('tanh'))
    return model


# Build a conditional GAN generator.
def build_cgan_generator(z_dim):
    z = layers.Input(shape=(z_dim,))
    # Condition label. The value ranges from 0 to 9
    label = layers.Input(shape=(1,), dtype='int32')
    # The conditional label is encoded with a tensor size of (batch size, 1, z_dim)
    label_embedding = layers.Embedding(num_classes, z_dim, input_length=1)(label)
    # The tensor is flattened and the size of the flattened tensor is (batch size, z_dim)
    label_embedding = layers.Flatten()(label_embedding)
    # Load the code into a random vector
    joined_representation = layers.Multiply()([z, label_embedding])
    generator = build_generator(z_dim)
    conditioned_img = generator(joined_representation)
    return models.Model([z, label], conditioned_img)


def build_discriminator(img_shape=(img_rows, img_cols, channels)):
    model = models.Sequential()
    model.add(
        layers.Conv2D(32,
                      kernel_size=3,
                      strides=2,
                      input_shape=(img_shape[0], img_shape[1], img_shape[2] + 1),
                      padding='same'))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(
        layers.Conv2D(64,
                      kernel_size=3,
                      strides=2,
                      input_shape=img_shape,
                      padding='same',
                      use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(
        layers.Conv2D(128,
                      kernel_size=3,
                      strides=2,
                      input_shape=img_shape,
                      padding='same',
                      use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Flatten())
    # This is binary
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# Construct a discriminator for conditional GAN.
def build_cgan_discriminator(img_shape):
    img = layers.Input(shape=img_shape)
    label = layers.Input(shape=(1,), dtype='int32')
    label_embedding = layers.Embedding(num_classes,
                                       np.prod(img_shape),
                                       input_length=1)(label)
    # The tensor is flattened and the size of the flattened tensor is (batch size, 28x28x1).
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape(img_shape)(label_embedding)
    # Here the image and type code are spliced to facilitate the subsequent calculation.
    concatenated = layers.Concatenate(axis=-1)([img, label_embedding])
    discriminator = build_discriminator(img_shape)
    # At the same time, the authenticity and category of images are judged.
    classification = discriminator(concatenated)
    return models.Model([img, label], classification)


def build_cgan(generator, discriminator):
    z = layers.Input(shape=(z_dim,))
    label = layers.Input(shape=(1,))
    img = generator([z, label])
    classification = discriminator([img, label])
    model = models.Model([z, label], classification)
    return model


# Build discriminators and generators.
discriminator = build_cgan_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])
generator = build_cgan_generator(z_dim)
# When the generator is trained, the parameters of the discriminator lock.
discriminator.trainable = False
cgan = build_cgan(generator, discriminator)
cgan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam())

losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):
    (X_train, y_train), (_, _) = datasets.mnist.load_data()
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)
    # All real data is labeled 1.
    real = np.ones((batch_size, 1))
    # All generated data is labeled 0.
    fake = np.zeros((batch_size, 1))
    for iteration in range(iterations):
        #  Training discriminator.
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        # Enter both the image and category here.
        imgs, labels = X_train[idx], y_train[idx]
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict([z, labels])
        d_loss_real = discriminator.train_on_batch([imgs, labels], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)
        #  Training generator.
        z = np.random.normal(0, 1, (batch_size, z_dim))
        g_loss = cgan.train_on_batch([z, labels], real)
        if (iteration + 1) % sample_interval == 0:
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration + 1, d_loss, 100.0 * accuracy, g_loss))
            sample_images()


train(10000, 32, 1000)

# Draw the resulting image.
image_grid_rows = 10
image_grid_columns = 5
z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
labels_to_generate = np.array([[i for j in range(5)] for i in range(10)])
labels_to_generate = labels_to_generate.flatten().reshape(-1, 1)
gen_imgs = generator.predict([z, labels_to_generate])
gen_imgs = 0.5 * gen_imgs + 0.5
fig, axs = plt.subplots(image_grid_rows,
                        image_grid_columns,
                        figsize=(10, 20),
                        sharey=True,
                        sharex=True)

cnt = 0
for i in range(image_grid_rows):
    for j in range(image_grid_columns):
        # Output a grid of images
        axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
        axs[i, j].axis('off')
        axs[i, j].set_title("Digit: %d" % labels_to_generate[cnt])
        cnt += 1
plt.show()