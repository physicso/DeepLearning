import numpy as np

# Common parameter.

# Image standardization parameter.
IMG_MEAN = np.array((179.506896209, 146.561481245, 181.496705501), dtype=np.float32)
NUM_CLASSES = 2
# Read 200x image layers.
USE_LEVEL = 1

# Preprocessing parameter.

LIST_DIRECTORY = 'CAMELYON/lists/'
POOL_SIZE = 8
IMAGE_DIRECTORY = 'CAMELYON/Train_Tumor/'
LABEL_DIRECTORY = 'CAMELYON/Ground_Truth/Mask/'
DATA_DIR = 'CAMELYON/images/'
DATA_LIST = 'CAMELYON/full_train_list'
TRAIN_SIZE = 320
WHOLE_THRESHOLD = 220 * 3

# Training parameter.

BATCH_SIZE = 8
INPUT_SIZE = '320, 320'
IMAGE_SIZE = int(INPUT_SIZE.split(', ')[0])
# Initial learning rate.
LEARNING_RATE = 1e-3
# Total iterations.
NUM_STEPS = 50000
MOMENTUM = 0.9
# The learning rate changes to the original POWER every iteration of LR_DECAY_STEP.
LR_DECAY_STEP = 10000
POWER = 0.5
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = 'snapshots_deeplabv3plus/'
WEIGHT_DECAY = 0.0005
# Whether to perform random scaling.
RANDOM_SCALE = True
# Whether to perform random mirroring and rotation.
RANDOM_MIRROR = True

# Test parameter.

DATA_DIRECTORY = 'CAMELYON/Test/'
PRED_DIRECTORY = 'preds/'
RESULT_DIRECTORY = 'result/'
DATA_LIST_PATH = 'CAMELYON/test_list'
# Predicted image size.
PREDEFINED_SIZE = 1000
# Consider the surroundings.
BORDER_SIZE = 100
# Environment information needs to be taken into account when predicting, so a circle of BORDER_SIZE is added to PREDEFINED_SIZE.
TEST_INPUT_SIZE = PREDEFINED_SIZE + 2 * BORDER_SIZE
# Simply filter the white background.
WHITE_THRESHOLD = 700
# Thumbnail ratio.
SCALE_RATIO = 20
RESTORE_FROM = 'snapshots_deeplabv3plus/model.ckpt-20000'
