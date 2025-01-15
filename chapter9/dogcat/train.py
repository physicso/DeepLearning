import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
from tensorflow.keras import layers, models
import data
import config
import resnet

# It is recommended to fix the randomized seeds in order to reproduce the results.
tf.set_random_seed(1234)


# The phased model parameters are stored.
def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)


x = layers.Input(shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNEL))
y_ = layers.Input(shape=(config.CLASS_NUM), )
global_step = tf.placeholder(dtype=tf.int32)
y = resnet.resnet_34(x)
model = models.Model(x, y)
print(model.summary())
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
lr = tf.train.exponential_decay(0.001, global_step, 5000, 0.5, staircase=True)
train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(cross_entropy)

# Build training data using data queues.
iterator, train_init_op = data.make_data(config.TRAIN_LIST, config.BATCH_SIZE)
data_iter = iterator.get_next()

# Perform basic settings of the TensorFlow environment.
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.allow_soft_placement = True
init = [tf.global_variables_initializer(), train_init_op]
saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1000)

logfile = open('resnet_34.log', 'w')
with tf.Session(config=tfconfig) as sess:
    # Initialize the graph.
    sess.run(init)
    for iteration in range(config.TRAIN_ITERATION):
        start_time = time.time()
        next_images, next_labels = sess.run(data_iter)
        # Use feed_dict to enter training data and labels.
        feed_dict = {x: next_images, y_: next_labels, global_step: iteration}
        # Run the calculation chart and output the result.
        pred, train_accuracy, train_loss, _ = sess.run([y, accuracy, cross_entropy, train_step], feed_dict=feed_dict)
        duration = time.time() - start_time
        logfile.write('Iteration: %5d,Accuracy: %.6f,Loss: %.6f,Duration: %.3f\n' % (
            iteration + 1, train_accuracy, train_loss, duration))
        if (iteration + 1) % config.SAVE_INTERVAL == 0:
            save(saver, sess, config.CHECKPOINT_DIR, iteration + 1)
            print('Iteration: %5d \t | Accuracy = %.6f, Loss = %.6f, (%.3f sec/iteration)' % (
                iteration + 1, train_accuracy, train_loss, duration))
            logfile.flush()
logfile.close()
