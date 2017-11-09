import tensorflow as tf

def initialize_weights(size):
    initial_value = tf.truncated_normal(size, 0.0, 0.001)
    return initial_value

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def disc_coef(i):
    
    return (tf.tanh((i - 10000)/15000) +1)/2
