import tensorflow as tf


def get_ds(x, y, f, batch_size):
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


def gray_scale(img):
    return tf.image.rgb_to_grayscale(img)
