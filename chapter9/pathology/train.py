import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import deeplabv3plus
from imagereader import ImageReader
import config

tf.set_random_seed(1234)


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


# The annotations are preprocessed.
def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size)
    input_batch = tf.squeeze(input_batch, squeeze_dims=[3])
    if one_hot:
        input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch


def main():
    h, w = map(int, config.INPUT_SIZE.split(','))
    input_size = (h, w)

    # Build data queue.
    coord = tf.train.Coordinator()
    reader = ImageReader(config.DATA_DIR, config.DATA_LIST, input_size, config.RANDOM_SCALE, config.RANDOM_MIRROR,
                         config.IMG_MEAN, coord)
    image_batch, label_batch, _ = reader.dequeue(config.BATCH_SIZE)
    raw_output = deeplabv3plus.deeplabv3plus(image_batch, config.NUM_CLASSES, [config.IMAGE_SIZE, config.IMAGE_SIZE])
    raw_output = tf.reshape(raw_output, [-1, config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUM_CLASSES])

    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]),
                               num_classes=config.NUM_CLASSES, one_hot=False)
    raw_gt = tf.reshape(label_proc, [-1, ])
    raw_output = tf.reshape(raw_output, [-1, config.NUM_CLASSES])

    # Remove meaningless classes (classes greater than config.NUM_CLASSES).
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, config.NUM_CLASSES - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_output, indices)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))
    step_ph = tf.placeholder(dtype=tf.int32)
    lr = tf.train.exponential_decay(config.LEARNING_RATE, step_ph, config.LR_DECAY_STEP, config.POWER, staircase=True)
    train_step = tf.train.MomentumOptimizer(lr, config.MOMENTUM).minimize(loss)
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    tfconfig.allow_soft_placement = True

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1000)
    logfile = open('deeplabv3plus.log', 'w')

    with tf.Session(config=tfconfig) as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        for step in range(config.NUM_STEPS):
            start_time = time.time()
            feed_dict = {step_ph: step}
            loss_value, _ = sess.run([loss, train_step], feed_dict=feed_dict)
            duration = time.time() - start_time
            logfile.write('Iteration: %5d,Loss: %.6f,Duration: %.3f\n' % (step + 1, loss_value, duration))
            if (step + 1) % config.SAVE_PRED_EVERY == 0:
                save(saver, sess, config.SNAPSHOT_DIR, step + 1)
                logfile.flush()
                print('Iteration: %5d \t | Loss: %.6f, (%.3f sec/step)' % ((step + 1), loss_value, duration))
        logfile.close()
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
