__author__ = "Nikhil Mehta"
__copyright__ = "--"
#---------------------------

import tensorflow as tf
import sys

from util import xavier_init, initialize_weights

class Adversarial_Generator:

    def __init__(self, config):
        self.trainable = True
        self.config = config

    def G (self, z, c, reuse=None):

        inputs = tf.concat(axis=1, values=[z, c])

        with tf.variable_scope('generator', reuse=reuse):

            G_h1 = self.fc_layer(inputs, self.config.z_dim+self.config.attr_dim, self.config.gh_dim, 'h1')
            G_h1 = tf.nn.relu(G_h1)
        
            G_out = self.fc_layer(G_h1, self.config.gh_dim, self.config.x_dim, 'out')
            # G_out = tf.nn.leaky_relu(G_out)
        
        return G_out

    def D (self, x, c, reuse=None):

        inputs = tf.concat(axis=1, values=[x, c])

        with tf.variable_scope('discriminator', reuse=reuse):

            D_h1 = self.fc_layer(inputs, self.config.x_dim+self.config.attr_dim, self.config.dh_dim, 'h1')
            D_h1 = tf.nn.leaky_relu(D_h1)

            D_logit = self.fc_layer(D_h1, self.config.dh_dim, 1, 'out')
            D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit

    def loss (self, x, z, c):

        x_gen = self.G(z, c)
        D_real, D_logit_real = self.D(x, c)
        # reuse
        D_fake, D_logit_fake = self.D(x_gen, c, reuse=True)

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        with tf.variable_scope('discriminator'):
            tf.summary.scalar("discriminator_loss", D_loss)

        with tf.variable_scope('generator'):
            tf.summary.scalar("generator_loss", G_loss)
        
        return G_loss, D_loss

    def fc_layer(self, input_data, in_size, out_size, name):

        with tf.variable_scope(name):
            W, b = self.get_fc_var(in_size, out_size, name)
            out = tf.matmul(input_data, W) + b
            
        return out

    def get_fc_var(self, in_size, out_size, name):

        initial_value = initialize_weights([in_size, out_size])
        #initial_value = xavier_init([in_size, out_size])
        W = tf.get_variable(name='weight', initializer=initial_value)

        initial_value = tf.zeros(shape=[out_size])
        b = tf.get_variable(name='bias', initializer=initial_value)

        return W, b

