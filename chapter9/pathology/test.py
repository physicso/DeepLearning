import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import numpy as np
from PIL import Image
from openslide import OpenSlide
import deeplabv3plus
import config


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def write(filename, content):
    new_image = Image.fromarray(np.uint8(content * 255.))
    new_image.save(filename, "PNG")


# The main function of this part of the function is to generate the position of the predicted image.
def generate_effective_regions(size):
    width = size[0]
    height = size[1]
    x_step = int(width / config.PREDEFINED_SIZE)
    y_step = int(height / config.PREDEFINED_SIZE)
    regions = []
    for x in range(0, x_step):
        for y in range(0, y_step):
            regions.append([x * config.PREDEFINED_SIZE, y * config.PREDEFINED_SIZE, 0, 0, config.PREDEFINED_SIZE - 1,
                            config.PREDEFINED_SIZE - 1])
    if not height % config.PREDEFINED_SIZE == 0:
        for x in range(0, x_step):
            regions.append([x * config.PREDEFINED_SIZE, height - config.PREDEFINED_SIZE, 0,
                            (y_step + 1) * config.PREDEFINED_SIZE - height, config.PREDEFINED_SIZE - 1,
                            config.PREDEFINED_SIZE - 1])
    if not width % config.PREDEFINED_SIZE == 0:
        for y in range(0, y_step):
            regions.append([width - config.PREDEFINED_SIZE, y * config.PREDEFINED_SIZE,
                            (x_step + 1) * config.PREDEFINED_SIZE - width, 0, config.PREDEFINED_SIZE - 1,
                            config.PREDEFINED_SIZE - 1])
    if not (height % config.PREDEFINED_SIZE == 0 or width % config.PREDEFINED_SIZE == 0):
        regions.append([width - config.PREDEFINED_SIZE, height - config.PREDEFINED_SIZE,
                        (x_step + 1) * config.PREDEFINED_SIZE - width, (y_step + 1) * config.PREDEFINED_SIZE - height,
                        config.PREDEFINED_SIZE - 1, config.PREDEFINED_SIZE - 1])
    return regions


# This part of the function is used to add a circle of environment to the generated images, which requires special processing of the boundary image.
def generate_overlap_tile(region, dimensions):
    shifted_region_x = region[0] - config.BORDER_SIZE
    shifted_region_y = region[1] - config.BORDER_SIZE
    clip_region_x = config.BORDER_SIZE
    clip_region_y = config.BORDER_SIZE
    if region[0] == 0:
        shifted_region_x = shifted_region_x + config.BORDER_SIZE
        clip_region_x = 0
    if region[1] == 0:
        shifted_region_y = shifted_region_y + config.BORDER_SIZE
        clip_region_y = 0
    if region[0] == dimensions[0] - config.PREDEFINED_SIZE:
        shifted_region_x = shifted_region_x - config.BORDER_SIZE
        clip_region_x = 2 * config.BORDER_SIZE
    if region[1] == dimensions[1] - config.PREDEFINED_SIZE:
        shifted_region_y = shifted_region_y - config.BORDER_SIZE
        clip_region_y = 2 * config.BORDER_SIZE
    return [shifted_region_x, shifted_region_y], [clip_region_x, clip_region_y]


def main():
    image = tf.placeholder(tf.float32, [config.TEST_INPUT_SIZE, config.TEST_INPUT_SIZE, 3])
    image_mean = (image - config.IMG_MEAN) / 255.

    image_batch = tf.expand_dims(image_mean, dim=0)

    raw_output = deeplabv3plus.deeplabv3plus(image_batch, config.NUM_CLASSES, [config.INPUT_SIZE, config.INPUT_SIZE])

    restore_var = tf.global_variables()

    # Here you need to output a probability graph.
    raw_output = raw_output[:, :, :, 1]
    prediction = tf.reshape(raw_output, [config.INPUT_SIZE, config.INPUT_SIZE])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, config.RESTORE_FROM)

    inference_list = open(config.DATA_LIST_PATH).readlines()
    for item in inference_list:
        image_name = item.split('\n')[0].split('/')[-1].split('.')[0]
        print('Diagnosing: ' + config.DATA_DIRECTORY + item.split('\n')[0])
        image_tif = OpenSlide(config.DATA_DIRECTORY + item.split('\n')[0])
        tif_dimensions = image_tif.level_dimensions[config.USE_LEVEL]
        regions = generate_effective_regions(tif_dimensions)
        index = 0
        region_num = len(regions)
        for region in regions:
            shifted_region, clip_region = generate_overlap_tile(region, tif_dimensions)
            index += 1
            if index % 1 == 0:
                print('  Progress: ' + str(index) + ' / ' + str(region_num))
            input_image = np.array(image_tif.read_region(location=(
            int(shifted_region[0] * image_tif.level_downsamples[config.USE_LEVEL]),
            int(shifted_region[1] * image_tif.level_downsamples[config.USE_LEVEL])), level=config.USE_LEVEL,
                                                         size=(config.TEST_INPUT_SIZE, config.TEST_INPUT_SIZE)))[:, :,
                          :-1]
            total_color_value = np.sum(input_image[clip_region[0]: (config.PREDEFINED_SIZE + clip_region[0]),
                                       clip_region[1]: (config.PREDEFINED_SIZE + clip_region[1])]) / (
                                            config.PREDEFINED_SIZE * config.PREDEFINED_SIZE)
            if total_color_value > config.WHITE_THRESHOLD:
                empty_prediction = np.zeros([config.PREDEFINED_SIZE, config.PREDEFINED_SIZE])[region[2]:(region[4] + 1),
                                   region[3]:(region[5] + 1)]
                write(config.PRED_DIRECTORY + image_name + '_' + str(region[0]) + '_' + str(
                    region[1]) + '_prediction.png', empty_prediction.astype(np.int8))
                continue
            prediction_result = sess.run(prediction, feed_dict={image: input_image})
            prediction_result = prediction_result[clip_region[0]: (config.PREDEFINED_SIZE + clip_region[0]),
                                clip_region[1]: (config.PREDEFINED_SIZE + clip_region[1])]
            prediction_result = prediction_result[region[2]:(region[4] + 1), region[3]:(region[5] + 1)]
            write(config.PRED_DIRECTORY + image_name + '_' + str(region[0]) + '_' + str(region[1]) + '_prediction.png',
                  prediction_result)


if __name__ == '__main__':
    main()
