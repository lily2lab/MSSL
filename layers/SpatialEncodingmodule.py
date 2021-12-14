import tensorflow as tf


class SpatialEncodingmodule():
    """Initialize the multiplicative unit.
    Args:
       layer_name: layer names for different multiplicative units.
       filter_size: int tuple of the height and width of the filter.
       num_hidden: number of units in output tensor.
    """
    def __init__(self, layer_name, num_hidden, filter_size):
        self.layer_name = layer_name
        self.num_features = num_hidden
        self.filter_size = filter_size

    def __call__(self, h, reuse=False):
        with tf.variable_scope(self.layer_name, reuse=reuse):
            g2 = tf.layers.conv2d(
                h, self.num_features, self.filter_size, padding='same', activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='g2_1')
            g2 = tf.layers.conv2d(
                g2, self.num_features, self.filter_size, padding='same', activation=tf.nn.leaky_relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='g2_2')

            g3 = tf.layers.conv2d(
                h, self.num_features, 1, padding='same', activation=tf.nn.leaky_relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='g3')
            mu =  g3+g2
            return mu
