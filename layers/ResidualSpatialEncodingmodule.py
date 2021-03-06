import tensorflow as tf
from layers.SpatialEncodingmodule import SpatialEncodingmodule as SE


class ResidualSpatialEncodingmodule():
    """Initialize the residual multiplicative block without mask.
    Args:
       layer_name: layer names for different residual multiplicative block.
       filter_size: int tuple of the height and width of the filter.
       num_hidden: number of units in output tensor.
    """
    def __init__(self, layer_name, num_hidden, filter_size):
        self.layer_name = layer_name
        self.num_features = num_hidden
        self.filter_size = filter_size

    def __call__(self, h, reuse=False):
        with tf.variable_scope(self.layer_name, reuse=reuse):
            h1 = tf.layers.conv2d(
                h, self.num_features, 1, padding='same', activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='h1')
            h2 = SE('SE_1', self.num_features, self.filter_size)(h1, reuse=reuse)
            h3 = SE('SE_2', self.num_features, self.filter_size)(h2, reuse=reuse)
            h4 = tf.layers.conv2d(
                h3, 2 * self.num_features, 1, padding='same', activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='h4')
            if h.shape[-1] is not h4.shape[-1]:
                h = tf.layers.conv2d(
                        h, 2* self.num_features, 1, padding='same', activation=None,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), name='res')
            rmb = tf.add(h, h4)
            return rmb
