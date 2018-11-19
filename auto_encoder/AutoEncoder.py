import keras
import tensorflow as tf
import additional_functions as af
from keras.datasets import cifar10
import numpy as np


class AutoEncoder(object):

    def residual(self, x, in_f, training, momentum=0.1, activation=tf.nn.relu):
        """
        It generates 4 convolutional layers to find edges 1-1 and residual connection between
        begin and end of block
        :param x: input_data
        :param in_f: number of block output filters
        :param momentum: momentum parameter to batch normalization between convolutional layer and activation
        :param activation: activation function
        :param training: boolean value to determine if it's the training phase
        :return: returns the output of the block
        """
        self.scope_num += 1
        with tf.variable_scope('model_mnist_res' + str(self.scope_num), reuse=tf.AUTO_REUSE):
            model = tf.layers.conv2d(x, in_f, (3, 1), 1, 'SAME')
            model = tf.layers.batch_normalization(model, momentum=momentum, training=training)
            model = activation(model)

            model = tf.layers.conv2d(model, in_f, (1, 3), 1, 'SAME')
            model = tf.layers.batch_normalization(model, momentum=momentum, training=training)
            model = activation(model)

            model = tf.layers.conv2d(model, in_f, (3, 1), 1, 'SAME')
            model = tf.layers.batch_normalization(model, momentum=momentum, training=training)
            model = activation(model)

            model = tf.layers.conv2d(model, in_f, (1, 3), 1, 'SAME')
            model = tf.layers.batch_normalization(model, momentum=momentum, training=training)
            model = activation(model)

            model = model + x
        return model

    def model(self, in_image, training):
        with tf.variable_scope('color', reuse=tf.AUTO_REUSE):
            model = self.residual(in_image, in_image.shape[-1], training)
        return model

    def train(self, num_epochs=10):
        """
        Method that trains the network
        using model method
        :param num_epochs: number of epochs
        """
        train_output = self.model(self.x_train, True)
        loss = self.loss_func(self.color_train, train_output)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = self.optimizer.minimize(loss)

        for i in range(num_epochs):
            self.sess.run(self.train_iterator.initializer)
            total_loss = []
            while True:
                try:
                    _, loss_value = self.sess.run([train_op, loss])
                    total_loss.append(loss_value)
                except tf.errors.OutOfRangeError:
                    break
            print('Total epoch {0} loss: {1}'.format(i, sum(total_loss) / len(total_loss)))

    def __init__(self, data_set=cifar10, batch_size=200, loss_func=tf.losses.mean_squared_error,
                 optimizer=tf.train.AdamOptimizer(0.01)):
        print("__init__")
        self.scope_num = 0  # used to label layers
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.optimizer = optimizer

        # Init data set
        train, test = data_set.load_data()
        self.train_iterator, (self.x_train, _, self.color_train) = af.get_ds(*train, f=af.gray_scale,
                                                                             batch_size=self.batch_size)
        self.test_iterator, (self.x_test, _, self.color_test) = af.get_ds(*test, f=af.gray_scale,
                                                                          batch_size=self.batch_size)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    pass
