from Tools.scripts import import_diagnostics


import importlib

importlib.import_module('models')
import matplotlib.pyplot as plt

batch_size = 200
salt_ratio = .05
pepper_ratio = .15


def get_ds(x, y, f):
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.shuffle(x.shape[0])
  ds = ds.batch(batch_size)
  it = ds.make_initializable_iterator()
  (images, labels) = it.get_next()
  images = tf.image.convert_image_dtype(images, tf.float32)
  if len(images.get_shape()) <= 3:
    images = tf.expand_dims(images, 3)
  additional = f(images)
  return it, (additional, labels, images)

def noise(img):
  random = tf.random_uniform(shape=tf.shape(img), minval=0.0, maxval=1.0)
  salt = tf.to_float(tf.greater_equal(random, 1.0 - salt_ratio))
  pepper = tf.to_float(tf.greater_equal(random, pepper_ratio))
  return tf.minimum(tf.maximum(img, salt), pepper)

def grayscale(img):
  return tf.image.rgb_to_grayscale(img)

def get_mnist():
  train, test = mnist.load_data()
  return get_ds(*train, noise), get_ds(*test, noise)

def get_cifar10():
  train, test = cifar10.load_data()
  return get_ds(*train, grayscale), get_ds(*test, grayscale)