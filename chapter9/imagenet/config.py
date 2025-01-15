# Preprocessing parameter.
DATA_DIR = 'Imagenet2012/ILSVRC2012_img_train'
CLASS_NUM = 10
TRAIN_LIST = 'train_tiny.csv'
TEST_LIST = 'test_imagenet.csv'
LABEL_MAPPING = 'label.csv'

# Data parameter.
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNEL = 3

# Training parameter.
BATCH_SIZE = 64
TRAIN_ITERATION = 100000
SAVE_INTERVAL = 1000
CHECKPOINT_DIR = 'resnet_34'

# Test parameter.
RESTORE_FROM = 'resnet_34/model.ckpt-30000'
MODEL_PREFIX = '%s/model.ckpt-%s'
