# Preprocessing parameter.
DATA_DIR = 'dog_vs_cat'
CLASS_NUM = 1
TRAIN_LIST = 'train_dogcat.csv'
TEST_LIST = 'test_dogcat.csv'
LABEL_MAPPING = 'label_dogcat.csv'

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
RESTORE_FROM = 'dogcat/resnet_34/model.ckpt-30000'
MODEL_PREFIX = '%s/model.ckpt-%s'
