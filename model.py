__author__ = "Nikhil Mehta"
__copyright__ = "--"
#---------------------------

import tensorflow as tf
import sys

from util import xavier_init

class Adversarial_Generator:

    def __init__(self, config):
        self.trainable = True
        self.config = config

    def build(self, x, z, c):

        self.x = x
        self.z = z
        self.c = c
        
        """ Discriminator Net model """
        
        self.D_W1 = tf.Variable(xavier_init([self.config.x_dim + self.config.attr_dim, self.config.dh_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.config.dh_dim]))

        self.D_W2 = tf.Variable(xavier_init([self.config.dh_dim, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

        theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

        """ Generator Net model """
        
        self.G_W1 = tf.Variable(xavier_init([self.config.z_dim + self.config.attr_dim, self.config.gh_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.config.gh_dim]))

        self.G_W2 = tf.Variable(xavier_init([self.config.gh_dim, self.config.x_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.config.x_dim]))

        theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        return theta_D, theta_G

    def loss (self):

        G_sample = self.generator(self.z, self.c)
        D_real, D_logit_real = self.discriminator(self.x, self.c)
        D_fake, D_logit_fake = self.discriminator(G_sample, self.c)

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        return G_loss, D_loss
        
    def generator(self, z, c):

        inputs = tf.concat(axis=1, values=[z, c])
        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.relu(G_log_prob)

        return G_prob

    def discriminator(self, x ,c):

        inputs = tf.concat(axis=1, values=[x, c])
        D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit

        
