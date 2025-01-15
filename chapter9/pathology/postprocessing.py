import glob
import numpy as np
from PIL import Image
import config


def image_to_array(input_image):
    im_array = np.array(input_image.getdata(), dtype=np.uint8)
    im_array = im_array.reshape((input_image.size[0], input_image.size[1]))
    return im_array


def write(filename, content):
    new_image = Image.fromarray(np.uint8(content))
    new_image.save(filename, "PNG")


prediction_list = glob.glob(config.PRED_DIRECTORY + '*_prediction.png')
image_list = {}
for prediction_image in prediction_list:
    name_parts = prediction_image.split('/')[-1].split('_')
    image_name, pos_x, pos_y = '_'.join(name_parts[:-3]), int(name_parts[-3]), int(name_parts[-2])
    if image_name in image_list:
        image_list[image_name].append([pos_x, pos_y])
    else:
        image_list[image_name] = []
        image_list[image_name].append([pos_x, pos_y])
for image_name in image_list.keys():
    print('Processing: ' + image_name)
    image_patches = []
    image_list[image_name].sort()
    last_x = -1
    row_patch = []
    for position in image_list[image_name]:
        pos_x = position[0]
        pos_y = position[1]
        image = Image.open(
            config.PRED_DIRECTORY + '_'.join([image_name, str(pos_x), str(pos_y), 'prediction']) + '.png')
        original_width, original_height = image.size
        if original_width < config.SCALE_RATIO or original_height < config.SCALE_RATIO:
            continue
        image = image.resize((int(original_width / config.SCALE_RATIO), int(original_height / config.SCALE_RATIO)),
                             Image.NEAREST)
        image_patch = image_to_array(image)
        if not pos_x == last_x:
            last_x = pos_x
            if len(row_patch) == 0:
                row_patch = image_patch
            else:
                if not len(image_patches) == 0:
                    image_patches = np.column_stack((image_patches, row_patch))
                else:
                    image_patches = row_patch
                row_patch = image_patch
        else:
            row_patch = np.row_stack((row_patch, image_patch))
    image_patches = np.column_stack((image_patches, row_patch))
    write(config.RESULT_DIRECTORY + '_'.join([image_name, 'prediction_thumbnail']) + '.png', image_patches)
    print('Prediction saved to ' + config.RESULT_DIRECTORY + '_'.join([image_name, 'prediction_thumbnail']) + '.png')
