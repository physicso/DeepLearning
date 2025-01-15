import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import config
import resnet


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


x = tf.placeholder(tf.float32, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNEL])
x_batch = tf.expand_dims(x, dim=0)
y = resnet.resnet_34(x_batch)

inference_list = open(config.TEST_LIST).readlines()

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.allow_soft_placement = True
init = tf.global_variables_initializer()
models = ['resnet-34']

with tf.Session(config=tfconfig) as sess:
    sess.run(init)
    restore_var = tf.global_variables()
    loader = tf.train.Saver(var_list=restore_var)
    for model in models:
        for index in range(config.SAVE_INTERVAL, config.TRAIN_ITERATION + 1, config.SAVE_INTERVAL):
            load(loader, sess, config.MODEL_PREFIX % (model, str(index)))
            if not os.path.exists(model):
                os.makedirs(model)
            result_file = open(model + '/' + str(index) + '.csv', 'w')
            for item in inference_list:
                filename, label = item.split('\n')[0].split(',')
                label = int(label)
                image = cv2.imread(config.DATA_DIR + '/' + filename).astype(np.float32)
                image = cv2.resize(image, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)) / 255.
                prediction = sess.run(y, feed_dict={x: image})
                result_file.write(filename + ',' + str(label) + ',' + str(prediction[0][1]) + '\n')
                # print(filename + ',' + str(label) + ',' + str(prediction[0][1]))
            result_file.flush()
            result_file.close
