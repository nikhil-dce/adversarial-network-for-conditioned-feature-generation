import argparse
import os, errno
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='3'

from data_handler import DataHandler
from model import Adversarial_Generator
from config import Config
from util import disc_coef

def main():

    parser = argparse.ArgumentParser(description='Adversarial Feature Generation')

    parser.add_argument('--iterations', type=int, default=1000000, help="iterations")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sample-generator', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--train-dir', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--z-dim', type=int, default=256)
    parser.add_argument('--gh-dim', type=int, default=512)
    parser.add_argument('--dh-dim', type=int, default=128)
    parser.add_argument('--g-steps', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    if not args.logdir and not args.sample_generator:
        print 'Please enter the log-dir name'
        sys.exit()
    
    if not args.train_dir or not os.path.exists(args.train_dir):
        raise IOError("Train Dir cannot be read")

    
    data_handler = DataHandler(args.train_dir)
    data_handler.load_train_data()
    #sys.exit()
    
    config = Config(args.batch_size, data_handler.x_dim, data_handler.attr_dim, args.z_dim, args.gh_dim, args.dh_dim, args.lr, args.g_steps)
    config.print_settings()
    
    model = Adversarial_Generator(config)

    x = tf.placeholder(tf.float32, shape=[None, config.x_dim])
    z = tf.placeholder(tf.float32, shape=[None, config.z_dim])
    c = tf.placeholder(tf.float32, shape=[None, config.attr_dim])
    step = tf.placeholder(tf.float32)
    
    theta_D, theta_G = model.build(x, z, c)
    G_loss, D_loss = model.loss()
    D_coef = disc_coef(step)
    D_loss_final = D_loss * D_coef
    D_solver = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(D_loss_final, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(G_loss, var_list=theta_G)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    session_config.allow_soft_placement = True

    sess = tf.Session(config=session_config)
    sess.run(tf.global_variables_initializer())

    i = 0
    train_num_batches = data_handler.train_size // config.batch_size
    data_index = 0

    for it in range(1000000):

        data_index += i*config.batch_size
        data_index %= train_num_batches
        x_batch, c_batch = data_handler.next_train_batch(data_index, config.batch_size)

        z_sample = sample_z(config.batch_size, config.z_dim)
        _, D_loss_curr, D_loss_f, D_cf = sess.run([D_solver, D_loss, D_loss_final, D_coef], feed_dict={x: x_batch, z: z_sample, c:c_batch, step:it})
        #D_loss_curr, D_loss_f, D_cf = sess.run([D_loss, D_loss_final, D_coef], feed_dict={x: x_batch, z: z_sample, c:c_batch, step:it})

        for gen_step in range(config.g_steps):
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={z: z_sample, c:c_batch})
            z_sample = sample_z(config.batch_size, config.z_dim)

        if it % 100 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('D loss Final: {:.4}'. format(D_loss_f))
            print('D Coef: {:.4}'. format(D_cf))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print('')

def sample_z(batch_size, z_dimen):
    return np.random.uniform(0., 1., size=[batch_size, z_dimen])

#return np.random.uniform(-1., 1., size=[batch_size, z_dimen])

if __name__ == '__main__':
    main()
    
