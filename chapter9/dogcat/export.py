import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import resnet
import config


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print('Restored model parameters from {}'.format(ckpt_path))


x = tf.placeholder(tf.float32, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH,
                                config.IMAGE_CHANNEL])
x_batch = tf.expand_dims(x, dim=0)
y = resnet.resnet_34(x_batch)

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.allow_soft_placement = True
init = tf.global_variables_initializer()

with tf.Session(config=tfconfig) as sess:
    sess.run(init)
    restore_var = tf.global_variables()
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, config.RESTORE_FROM)
    # Note the "code" in the following code: input and prediction
    tf.saved_model.simple_save(sess,
                               "./dogcat/00000001",
                               inputs={"input": x},
                               outputs={"prediction": y})
