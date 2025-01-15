import os
import glob
from multiprocessing import Pool
import numpy as np
from PIL import Image
from openslide import OpenSlide
import config


# Each process independently processes the slide at hand.
def process_image(image_name):
    filename = image_name.split('/')[-1]
    print('Processing: ' + filename)
    # To implement asynchronous processing, you need to place the list in a folder.
    if os.path.exists(config.LIST_DIRECTORY + 'train_list_' + filename.split('.')[0]):
        os.remove(config.LIST_DIRECTORY + 'train_list_' + filename.split('.')[0])
    train_list = open(config.LIST_DIRECTORY + 'train_list_' + filename.split('.')[0], 'w')
    label_filename = config.LABEL_DIRECTORY + 'Tumor_' + filename.split('.')[0].split('_')[-1] + '_Mask.tif'
    # Open the slide and its corresponding label.
    image_tif = OpenSlide(image_name)
    label_tif = OpenSlide(label_filename)

    # Get the size of the slide.
    image_width = image_tif.level_dimensions[config.USE_LEVEL][0]
    image_height = image_tif.level_dimensions[config.USE_LEVEL][1]
    width_step_num = int(image_width / config.TRAIN_SIZE)
    height_step_num = int(image_height / config.TRAIN_SIZE)

    # Read images sequentially.
    for x_index in range(width_step_num):
        for y_index in range(height_step_num):
            image = np.array(image_tif.read_region(
                location=(int(x_index * config.TRAIN_SIZE * image_tif.level_downsamples[config.USE_LEVEL]),
                          int(y_index * config.TRAIN_SIZE * image_tif.level_downsamples[config.USE_LEVEL])),
                level=config.USE_LEVEL, size=(config.TRAIN_SIZE, config.TRAIN_SIZE)))[:, :, :-1]
            total_color_value = np.sum(image)
            # Filter out white background.
            if total_color_value > config.WHOLE_THRESHOLD:
                continue
            label = np.array(label_tif.read_region(
                location=(int(x_index * config.TRAIN_SIZE * image_tif.level_downsamples[config.USE_LEVEL]),
                          int(y_index * config.TRAIN_SIZE * image_tif.level_downsamples[config.USE_LEVEL])),
                level=config.USE_LEVEL, size=(config.TRAIN_SIZE, config.TRAIN_SIZE)))[:, :, :-1]
            lb = Image.new('L', (config.TRAIN_SIZE, config.TRAIN_SIZE), 0)
            new_label = lb.load()
            for i in range(config.TRAIN_SIZE):
                for j in range(config.TRAIN_SIZE):
                    # In the original labels, the cancer pixel is 255, which needs to be converted to 1.
                    if int(label[i, j, 0]) == 255:
                        new_label[j, i] = 1
            # Save images and corresponding labels, and write to the training list.
            im = Image.fromarray(image, 'RGB')
            image_outfile = filename.split('.')[0] + '_' + str(x_index) + '_' + str(y_index) + '.png'
            label_outfile = filename.split('.')[0] + '_' + str(x_index) + '_' + str(y_index) + '_label.png'
            im.save(config.DATA_DIR + image_outfile)
            lb.save(config.DATA_DIR + label_outfile)
            train_list.write(image_outfile + ' ' + label_outfile + '\n')
        train_list.flush()
    train_list.close()


image_list = glob.glob(config.IMAGE_DIRECTORY + '*.tif')
# Use multiple processes for asynchronous processing.
process_pool = Pool(config.POOL_SIZE)
res = process_pool.map_async(process_image, image_list)
res.get()
