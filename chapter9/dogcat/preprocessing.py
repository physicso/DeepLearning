import glob
import random
import config

with open(config.TRAIN_LIST, 'w') as train_file:
    with open(config.TEST_LIST, 'w') as test_file:
        print(config.TRAIN_LIST)
        print(config.LABEL_MAPPING)
        for index, path in enumerate(glob.glob(config.DATA_DIR + '/*')):
            print(str(index) + ' ' + path)
            img = path.split('/')[-1]
            label = int(img.split('.')[0] == 'dog')
            if random.random() >= 0.2:
                train_file.write(img + ',' + str(label) + '\n')
            else:
                test_file.write(img + ',' + str(label) + '\n')
