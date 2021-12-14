import tensorflow as tf
from layers.SpatialEncodingmodule import SpatialEncodingmodule as SE


class S2M():
    """Initialize the causal multiplicative unit.
    Args:
       layer_name: layer names for different causal multiplicative unit.
       filter_size: int tuple of the height and width of the filter.
       num_hidden: number of units in output tensor.
    """
    def __init__(self, layer_name, num_hidden, filter_size, Tu_length):
        self.layer_name = layer_name
        self.num_features = num_hidden
        self.filter_size = filter_size
        self.Tu_length = Tu_length

    def __call__(self, h1, h2, stride=False, reuse=False):
        with tf.variable_scope(self.layer_name, reuse=reuse):
            hl = h1
            hr = h2
            for i in range(2*self.Tu_length):
                hl = SE('previous_frame_1'+str(i), self.num_features, self.filter_size)(hl, reuse=reuse)
            h10 = tf.layers.conv2d(
                h1, self.num_features,1, padding='same', activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='previous_l')
            hl = hl+h10
 
            for i in range(self.Tu_length):
                hr = SE('previous_frame_2'+str(i), self.num_features, self.filter_size)(hr, reuse=reuse)
            h20 = tf.layers.conv2d(
                h2, self.num_features,1, padding='same', activation=None, # tf.nn.leaky_relu
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='previous_r')
            hr = hr+h20

            h_diff = tf.subtract(hr, hl)
            for i in range(self.Tu_length):
                h_diff = SE('motion_layer'+str(i), self.num_features, self.filter_size)(h_diff, reuse=reuse)
            h2 = tf.layers.conv2d(
                h2, self.num_features,1, padding='same', activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='align')

            h = tf.add(h_diff,h2)
            for i in range(self.Tu_length):
                h = SE('fusion_layer'+str(i), self.num_features, 1)(h, reuse=reuse)

            return h
