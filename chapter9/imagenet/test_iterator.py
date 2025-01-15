import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import config
import resnet
import data


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print('Restored model parameters from {}'.format(ckpt_path))


x = tf.placeholder(tf.float32, [None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNEL])
y_ = tf.placeholder(tf.float32, [None, config.CLASS_NUM])
y = resnet.resnet_34(x)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))

test_iterator, test_init_op = data.make_data(config.TEST_LIST, config.BATCH_SIZE, is_train=False)
test_data_iter = test_iterator.get_next()

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.allow_soft_placement = True
init = [tf.global_variables_initializer(), test_init_op]

with tf.Session(config=tfconfig) as sess:
    sess.run(init)
    restore_var = tf.global_variables()
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, config.RESTORE_FROM)
    accuracies = []
    losses = []
    for _ in range(int(5000 / config.BATCH_SIZE)):
        next_test_images, next_test_labels = sess.run(test_data_iter)
        feed_dict = {x: next_test_images, y_: next_test_labels}
        test_accuracy, test_loss = sess.run([accuracy, cross_entropy], feed_dict=feed_dict)
        accuracies.append(test_accuracy)
        losses.append(test_loss)
    print('Test Accuracy = %.6f, Test Loss = %.6f' % (sum(accuracies) / len(accuracies), sum(losses) / len(losses)))
