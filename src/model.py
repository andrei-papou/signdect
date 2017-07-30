from typing import Iterable
import numpy as np
import tensorflow as tf
from models.standard import standard, standard_deep, standard_simplified
from models.sign_specific import sign_specific
from utils import generate_model_id, get_model_file_path, get_mode_prefix
from data import DataProvider
from settings import Mode
from exceptions import EarlyStoppingException


class Model:

    def __init__(self, width, height, channels, categories, stop_in=1,
                 model_id=None, saved_model=None, var_scope='signdect_model_scope'):
        self.width = width
        self.height = height
        self.channels = channels
        self.categories = categories
        self.var_scope = var_scope
        self.net_vars_created = None
        self.model_path = get_model_file_path(model_id or generate_model_id())
        self.saved_model_path = get_model_file_path(saved_model) if saved_model is not None else None
        self.best_accuracy = 0
        self.stop_in = stop_in
        self.epochs_no_improvement = 0

    def _get_output_ten(self, inputs_ph):
        with tf.variable_scope(self.var_scope, reuse=self.net_vars_created):
            if self.net_vars_created is None:
                self.net_vars_created = True

            inputs = tf.reshape(inputs_ph, [-1, self.width, self.height, self.channels])
            net = sign_specific(inputs, self.categories)
            net = tf.check_numerics(net, message='model')
        return net

    def _get_loss_op(self, output, labels):
        one_hot_labels = tf.one_hot(tf.cast(labels, dtype=tf.int32), depth=self.categories)
        return tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=one_hot_labels)

    @staticmethod
    def _get_accuracy_op(output, labels):
        return tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output, axis=1), labels), dtype=tf.int32))

    def _monitor(self, sess: tf.Session, batches_generator: Iterable, mini_batch_size: int, mode: Mode):
        input_ph = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, self.width, self.height, self.channels))
        labels_ph = tf.placeholder(dtype=tf.int64, shape=(mini_batch_size,))
        num_batches, loss, correctly_predicted, data_set_size = 0, 0, 0, 0

        output = self._get_output_ten(input_ph)
        accuracy_ten = self._get_accuracy_op(output, labels_ph)
        batch_loss = tf.reduce_sum(self._get_loss_op(output, labels_ph))

        for xs, ys in batches_generator:
            feed_dict = {input_ph: xs, labels_ph: ys}
            correctly_predicted += sess.run(accuracy_ten, feed_dict=feed_dict)
            loss += sess.run(batch_loss, feed_dict=feed_dict)
            num_batches += 1
            data_set_size += len(ys)

        prefix = get_mode_prefix(mode)
        print('{} loss:       {}'.format(prefix, loss / num_batches))
        accuracy = correctly_predicted / data_set_size
        print('{} accuracy:   {}'.format(prefix, accuracy))
        print('')  # new line

        if mode == Mode.TEST:
            if accuracy > self.best_accuracy:
                self._save(sess)
                self.best_accuracy = accuracy
                self.epochs_no_improvement = 0
            else:
                self.epochs_no_improvement += 1
                print('No improvement in {} / {} epochs. Best accuracy: {}'
                      .format(self.epochs_no_improvement, self.stop_in, self.best_accuracy))
                if self.epochs_no_improvement >= self.stop_in:
                    raise EarlyStoppingException

    def _load_or_initialize(self, sess):
        if self.saved_model_path is None:
            sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(sess, self.saved_model_path)

    def _save(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.model_path)

    def train(self, data_provider: DataProvider, num_epochs, mini_batch_size, learning_rate):
        input_ph = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, self.width, self.height, self.channels))
        labels_ph = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size,))
        output = self._get_output_ten(input_ph)

        loss = self._get_loss_op(output, labels_ph)
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.Session() as sess:
            self._load_or_initialize(sess)

            for ep in range(num_epochs):
                for xs, ys in data_provider.get_batches(mode=Mode.TRAIN):
                    feed_dict = {input_ph: xs, labels_ph: ys}
                    sess.run(train_op, feed_dict=feed_dict)

                print('Epoch {} training complete'.format(ep))
                self._monitor(
                    sess=sess,
                    batches_generator=data_provider.get_batches(mode=Mode.TRAIN),
                    mini_batch_size=mini_batch_size,
                    mode=Mode.TRAIN)
                self._monitor(
                    sess=sess,
                    batches_generator=data_provider.get_batches(mode=Mode.TEST),
                    mini_batch_size=mini_batch_size,
                    mode=Mode.TEST)

    def infer(self, img: np.array) -> int:
        input_ph = tf.placeholder(dtype=tf.float32, shape=(self.width, self.height, self.channels))
        category_ten = tf.argmax(self._get_output_ten(input_ph), axis=1)
        with tf.Session() as sess:
            self._load_or_initialize(sess)
            category = sess.run(category_ten, feed_dict={input_ph: img})
        return category[0]
