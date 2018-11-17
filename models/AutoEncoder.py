import keras
from keras.datasets import cifar10, mnist
import tensorflow as tf
import numpy as np


class AutoEncoder(object):

    def residual(self, x, in_f, momentum, activation, training):
        """
        It generates 4 convolution layers to find edges 1-1 and residual connection between
        begin and end of block
        :param in_f: number of block output filters
        :param momentum: momentum parameter to barch normalization between conv and activation
        :param activation: activation function
        :param training: boolean value to determine if it's the training phase
        :return: returns the output of the block
        """
        self.scope_num  +=1
        with tf.variable_scope('model_mnist_res' + str(self.scope_num ), reuse=tf.AUTO_REUSE):
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


    def model(self):
        """
        Method that generates the model
        """
        print('Lets start generating model')

    def __init__(self):
        self.scope_num = 0 #used to label layers
        self.model()

    pass


AutoEncoder