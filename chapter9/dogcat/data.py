import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import config


# Each piece of data read is processed.
def _read_list(record):
    # The training data consists of file names and labels in CSV format
    image_path, label = tf.io.decode_csv(record,
                                         record_defaults=[[''], [0]])
    label = tf.cast(label, tf.int32)
    onehot_label = tf.one_hot(label, config.CLASS_NUM)
    return image_path, onehot_label


# Each image is processed.
def _process_images(image_path, label):
    # Read and decode the image file.
    filename = config.DATA_DIR + '/' + image_path
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=config.IMAGE_CHANNEL)
    image_decoded = tf.cast(image_decoded, tf.float32) / 255.
    # Rensize the image.
    image_resized = tf.image.resize(image_decoded, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH])
    # Random horizontal mirror image.
    image_flipped = tf.image.random_flip_left_right(image_resized)
    # Random brightness.
    image_distorted = tf.image.random_brightness(image_flipped, max_delta=60)
    # Random contrast.
    image_distorted = tf.image.random_contrast(image_distorted, lower=0.2, upper=2.0)
    # Image standardization.
    image_distorted = tf.image.per_image_standardization(image_distorted)
    return image_distorted, label


def make_data(file, batch_size, is_train=True):
    dataset = tf.data.TextLineDataset(file)
    if is_train:
        # Randomize the data, using buffersize as the size of the cache pool.
        dataset = dataset.shuffle(buffer_size=10000)
    # For each piece of data read, _read_list and _process_images are used for processing.
    dataset = dataset.map(_read_list, num_parallel_calls=4)
    dataset = dataset.map(_process_images, num_parallel_calls=4)
    # Merge read data based on the batch_size size.
    dataset = dataset.batch(batch_size=config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_train:
        # Repeat the data list multiple times (more than the number of training rounds).
        dataset = dataset.repeat(100)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    train_init_op = iterator.make_initializer(dataset)
    return iterator, train_init_op
