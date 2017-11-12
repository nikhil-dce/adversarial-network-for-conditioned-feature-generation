__author__ = "Nikhil Mehta"
__copyright__ = "--"
#---------------------------

import tensorflow as tf
import numpy as np
import sys

from util import xavier_init, initialize_weights

class Adversarial_Generator:

    def __init__(self, config):
        self.trainable = True
        self.config = config

    def G (self, z, c, reuse=None, is_training=True):

        with tf.variable_scope('generator', reuse=reuse):

            inputs = tf.concat(axis=1, values=[z, c])

            G_h1 = self.fc_layer(inputs, self.config.z_dim+self.config.attr_dim, self.config.gh1_dim, 'h1')
            G_h1 = tf.nn.relu(G_h1)
            #G_h1 = batch_norm(name='bn_h1')(G_h1, phase=is_training)

            G_h2 = self.fc_layer(G_h1, self.config.gh1_dim, self.config.gh2_dim, 'h2')
            G_h2 = tf.nn.relu(G_h2)
            #G_h2 = batch_norm(name='bn_h2')(G_h2, phase=is_training)
            
            G_out = self.fc_layer(G_h2, self.config.gh2_dim, self.config.x_dim, 'out')
            G_out = tf.tanh(G_out)
                    
        return G_out

    def D (self, x, c, reuse=None, bn_name='same'):
        """ bn_name useful for batch_norm. Separate batch for real/fake"""
        with tf.variable_scope('discriminator', reuse=reuse) as d_scope:

            #x = self.dropout(x, 0.5, name="x_dropout")
            inputs = tf.concat(axis=1, values=[x, c])
                                    
            D_h1 = self.fc_layer(inputs, self.config.x_dim+self.config.attr_dim, self.config.dh1_dim, 'h1')
            D_h1 = tf.nn.leaky_relu(D_h1)
            #D_h1 = batch_norm(name='bn_h1')(D_h1)

            #D_h2 = self.fc_layer(D_h1, self.config.dh1_dim, self.config.dh2_dim, 'h2')
            #D_h2 = tf.nn.leaky_relu(D_h2)
            #D_h2 = batch_norm(name='bn_h2')(D_h2)
            
            D_logit = self.fc_layer(D_h1, self.config.dh1_dim, 1, 'out')
            D_prob = tf.nn.sigmoid(D_logit)
       
        return D_prob, D_logit

    def loss (self, x, z, c):

        x_gen = self.G(z, c)

        D_real, D_logit_real = self.D(x, c, bn_name='real')
        # reuse
        D_fake, D_logit_fake = self.D(x_gen, c, reuse=True, bn_name='fake')
        reverse_D_real, reverse_D_logit_real = self.D(x_gen, c, reuse=True, bn_name='fake')
        reverse_D_fake, reverse_D_logit_fake = self.D(x, c, reuse=True, bn_name='real')
        
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake

        reverse_D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=reverse_D_logit_real, labels=tf.ones_like(reverse_D_logit_real)))
        reverse_D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=reverse_D_logit_fake, labels=tf.zeros_like(reverse_D_logit_fake)))
        reverse_D_loss = reverse_D_loss_real + reverse_D_loss_fake
        
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        with tf.variable_scope('discriminator'):
            tf.summary.scalar("discriminator_loss", D_loss)
            
        with tf.variable_scope('generator'):
            tf.summary.scalar("generator_loss", G_loss)
        
        return G_loss, D_loss, reverse_D_loss

    # ------------------WGAN------------------------------
    def D_WGAN (self, x, c, reuse=None, bn_name='same'):
        """ bn_name useful for batch_norm. Separate batch for real/fake"""
        with tf.variable_scope('discriminator', reuse=reuse) as d_scope:

            inputs = tf.concat(axis=1, values=[x, c])
                                    
            D_h1 = self.fc_layer(inputs, self.config.x_dim+self.config.attr_dim, self.config.dh1_dim, 'h1')
            D_h1 = tf.nn.leaky_relu(D_h1)
                        
            D_logit = self.fc_layer(D_h1, self.config.dh1_dim, 1, 'out')

        return D_logit

    def loss_WGAN (self, x, z, c):

        x_gen = self.G(z, c)

        D_logit_real = self.D_WGAN(x, c, bn_name='real')
        # reuse
        D_logit_fake = self.D_WGAN(x_gen, c, reuse=True, bn_name='fake')

        D_loss = tf.reduce_mean(D_logit_fake) - tf.reduce_mean(D_logit_real) 
        G_loss = - tf.reduce_mean(D_logit_fake)

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

    def dropout(self, bottom, keep_prob, name):
          return tf.nn.dropout(bottom, keep_prob, name=name)



class batch_norm(object):
     
    #def __init__(self, epsilon=1e-5, momentum = 0.99, name="batch_norm"):
    def __init__(self, name="batch_norm"):
          with tf.variable_scope(name):
               #self.epsilon  = epsilon
               #self.momentum = momentum
               self.name = name


    # Keep phase True for both train and test case for GANs
    def __call__(self, x, phase=True):
          return tf.contrib.layers.batch_norm(x,
                                              #decay=self.momentum, 
                                              #epsilon=self.epsilon,
                                              scale=True,
                                              center=True, 
                                              is_training=phase,
                                              scope=self.name)
