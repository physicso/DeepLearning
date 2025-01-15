import glob
import config

labels = []
imgs = [[] for _ in range(config.CLASS_NUM)]

with open(config.TRAIN_LIST, 'w') as train_file:
    with open(config.LABEL_MAPPING, 'w') as label_file:
        print(config.TRAIN_LIST)
        print(config.LABEL_MAPPING)
        for index, path in enumerate(glob.glob(config.DATA_DIR + '/*')):
            print(str(index) + ' ' + path)
            label = path.split('/')[-1]
            print(label)
            label_file.write(str(index) + ',' + label + '\n')
            labels.append(label)
            for img_path in glob.glob(config.DATA_DIR + '/' + label + '/*'):
                img = img_path.split('/')[-1]
                train_file.write(label + '/' + img + ',' + str(index) + '/n')
                imgs[index].append(img)
            train_file.flush()
            label_file.flush()
