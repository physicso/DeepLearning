import os

os.environ["CUDA_VISIBLE_DEVICES"] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import deeplabv3plus


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    image = tf.placeholder(tf.float32, [None, None, 3])
    image_mean = (image - config.IMG_MEAN) / 255.
    image_batch = tf.expand_dims(image_mean, dim=0)
    raw_output = deeplabv3plus.deeplabv3plus(image_batch, config.NUM_CLASSES,
                                             [config.TEST_INPUT_SIZE, config.TEST_INPUT_SIZE])
    prediction = raw_output[0, :, :, 1]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        restore_var = tf.global_variables()
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, config.RESTORE_FROM)
        tf.saved_model.simple_save(sess, 'pathology/0000000001', inputs={'input': image},
                                   outputs={'prediction': prediction})


if __name__ == '__main__':
    main()
