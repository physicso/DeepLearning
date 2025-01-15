import random
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


# Random scaling.
def image_scaling(img, label, w, h):
    scale = random.uniform(1.0, 1.25)
    w_new = int(w * scale)
    h_new = int(h * scale)
    new_shape = tf.convert_to_tensor([w_new, h_new])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_images(label, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    st_w = w_new - w
    st_h = h_new - h

    tmp = int(random.random() * (st_w + 1))
    img = img[tmp: tmp + w - 1, :, :]
    label = label[tmp: tmp + w - 1, :, :]

    tmp = int(random.random() * (st_h + 1))
    img = img[:, tmp:tmp + h - 1, :]
    label = label[:, tmp:tmp + h - 1, :]
    img = tf.image.resize_images(img, tf.convert_to_tensor([w, h]))
    label = tf.image.resize_images(label, tf.convert_to_tensor([w, h]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img, label


# Random mirroring and rotation.
def image_mirroring(img, label):
    if random.random() >= 0.5:
        img = tf.image.flip_left_right(img)
        label = tf.image.flip_left_right(label)
    if random.random() >= 0.5:
        img = tf.image.flip_up_down(img)
        label = tf.image.flip_up_down(label)
    if random.random() >= 0.5:
        img = tf.image.rot90(img, 1)
        label = tf.image.rot90(label, 1)
    if random.random() >= 0.5:
        img = tf.image.rot90(img, 2)
        label = tf.image.rot90(label, 2)
    if random.random() >= 0.5:
        img = tf.image.rot90(img, 3)
        label = tf.image.rot90(label, 3)
    return img, label


# Read list.
def read_labeled_image_list(data_dir, data_list):
    f = open(data_list, 'r')
    images = []
    labels = []
    names = []
    for line in f:
        try:
            if len(line.strip("\n").split(' ')) == 2:
                image, label = line.strip("\n").split(' ')
            elif len(line.strip("\n").split(' ')) == 1:
                image = line.strip("\n")
                label = ''
            name = image.split('.')[0]
        except ValueError:
            print('List file format wrong!')
            exit(1)
        images.append(data_dir + image)
        labels.append(data_dir + label)
        names.append(name)
    return images, labels, names


# Read images and labels.
def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, img_mean):
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    name_contents = input_queue[2]

    img = tf.image.decode_png(img_contents, channels=3)
    img = tf.cast(img, dtype=tf.float32)

    # Subtract the mean to balance data distribution.
    img -= img_mean
    label = tf.image.decode_png(label_contents, channels=1)

    if input_size is not None:
        w, h = input_size
        img = tf.image.resize_images(img, tf.convert_to_tensor([w, h]))
        label = tf.image.resize_images(label, tf.convert_to_tensor([w, h]),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if random_scale:
            img, label = image_scaling(img, label, w, h)

        if random_mirror:
            img, label = image_mirroring(img, label)

    return img, label, tf.convert_to_tensor(name_contents, dtype=tf.string)


class ImageReader(object):
    def __init__(self, data_dir, data_list, input_size, random_scale, random_mirror, ignore_label, img_mean, coord):
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.label_list, self.name_list = read_labeled_image_list(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.image_names = tf.convert_to_tensor(self.name_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels, self.image_names],
                                                   shuffle=input_size is not None)
        self.image, self.label, self.img_name = read_images_from_disk(self.queue, self.input_size, random_scale,
                                                                      random_mirror, ignore_label, img_mean)

    def dequeue(self, num_elements):
        image_batch, label_batch, name_batch = tf.train.shuffle_batch([self.image, self.label, self.img_name],
                                                                      num_elements, num_elements * 4, num_elements * 2)
        return image_batch, label_batch, name_batch
