__author__ = "Nikhil Mehta"
__copyright__ = "--"
#---------------------------

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
from linear_svm import LinearSVM

from config import Config
from util import disc_coef

def initialize_summary_writer(logdir, sess):

    RUN_DIR = "run_1"
    logdir = os.path.join(logdir, RUN_DIR)

    print ("LOG Directory is %s" % (logdir))

    if tf.gfile.Exists(logdir):
        print 'Deleting existing data in ' + logdir
        tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)

    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    meta_graph_def = tf.train.export_meta_graph(filename=logdir+'/my-model.meta')

    return summary_writer

def main():

    parser = argparse.ArgumentParser(description='Adversarial Feature Generation')

    parser.add_argument('--iterations', type=int, default=1000000, help="iterations")
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--sample-generator', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--train-dir', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--z-dim', type=int, default=100)
    parser.add_argument('--gh-dim', type=int, default=128)
    parser.add_argument('--dh-dim', type=int, default=256)
    parser.add_argument('--g-steps', type=int, default=1)
    parser.add_argument('--d-steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()

    if not args.logdir and not args.sample_generator:
        print 'Please enter the log-dir name'
        sys.exit()
    
    if not args.train_dir or not os.path.exists(args.train_dir):
        raise IOError("Train Dir cannot be read")

    data_handler = DataHandler(args.train_dir)
    data_handler.load_train_data()
        
    config = Config(args.batch_size, data_handler.x_dim, data_handler.attr_dim, args.z_dim, args.gh_dim, args.dh_dim, args.lr, args.g_steps, args.d_steps)
    config.print_settings()
    
    model = Adversarial_Generator(config)

    x = tf.placeholder(tf.float32, shape=[None, config.x_dim])
    z = tf.placeholder(tf.float32, shape=[None, config.z_dim])
    c = tf.placeholder(tf.float32, shape=[None, config.attr_dim])
    step = tf.placeholder(tf.float32)
        
    G_loss, D_loss = model.loss(x,z,c)
    #D_coef = disc_coef(step)
    #D_loss_final = D_loss * D_coef

    theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    g_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='generator')
    d_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='discriminator')
    #d_summaries.append(tf.summary.scalar('disc_coef', D_coef))

    # D_optimization
    D_solver = tf.train.AdamOptimizer(learning_rate=args.lr)
    grads = D_solver.compute_gradients(D_loss, var_list=theta_D)

    for grad, var in grads:
            if grad is not None:
                d_summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
                
    d_apply_gradients = D_solver.apply_gradients(grads, name='apply_disc_gradients')

    D_parameters = 0
    for var in theta_D:
        d_summaries.append(tf.summary.histogram(var.op.name, var))
        # Calculate total parameters
        shape = var.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
            D_parameters += variable_parameters

    print 'Parameters D: %d' % D_parameters
    
    # G_optimization
    G_solver = tf.train.AdamOptimizer(learning_rate=args.lr)
    grads = D_solver.compute_gradients(G_loss, var_list=theta_G)

    for grad, var in grads:
            if grad is not None:
                g_summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
                
    g_apply_gradients = G_solver.apply_gradients(grads, name='apply_gen_gradients')

    G_parameters = 0
    for var in theta_G:
        g_summaries.append(tf.summary.histogram(var.op.name, var))
        # Calculate total parameters
        shape = var.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
            G_parameters += variable_parameters

    print 'Parameters G: %d' % G_parameters         
    
    g_summary_op = tf.summary.merge(g_summaries)
    d_summary_op = tf.summary.merge(d_summaries)

    # Accuracy Summary
    acc_val = tf.placeholder(tf.float32)
    summ_op = tf.summary.scalar('accuracy_score', acc_val)
    
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    session_config.allow_soft_placement = True

    sess = tf.Session(config=session_config)
    sess.run(tf.global_variables_initializer())

    summary_writer = initialize_summary_writer(args.logdir, sess)

    print ('------------------GRAPH is SAVED------------')

    train_num_batches = data_handler.train_size // config.batch_size

    for it in range(10000000):

        batch_index = it % train_num_batches
        data_index = batch_index*config.batch_size
        x_batch, c_batch = data_handler.next_train_batch(data_index, config.batch_size)

        for disc_step in range(config.d_steps):
            z_sample = sample_z(config.batch_size, config.z_dim)
            total_disc_steps = disc_step + config.d_steps*it
            if total_disc_steps % 500 == 0:
                _, D_loss_curr, D_summary = sess.run([d_apply_gradients, D_loss, d_summary_op], feed_dict={x: x_batch, z: z_sample, c:c_batch, step:it})
                summary_writer.add_summary(D_summary, total_disc_steps)
            else:   
                _, D_loss_curr = sess.run([d_apply_gradients, D_loss], feed_dict={x: x_batch, z: z_sample, c:c_batch, step:it})
            
        for gen_step in range(config.g_steps):
            z_sample = sample_z(config.batch_size, config.z_dim)
            total_gen_steps = gen_step + config.g_steps*it
            if total_gen_steps % 500 == 0:
                _, G_loss_curr, G_summary = sess.run([g_apply_gradients, G_loss, g_summary_op], feed_dict={z: z_sample, c:c_batch})
                summary_writer.add_summary(G_summary, total_gen_steps)
            else:
                _, G_loss_curr = sess.run([g_apply_gradients, G_loss], feed_dict={z: z_sample, c:c_batch})
                
        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print('')

        if it % 50000 == 0:
            summ = test_accuracy(sess, summ_op, acc_val, model, data_handler, config)
            summary_writer.add_summary(summ, it)

def test_accuracy(sess, summ_op, acc_val, model, data_handler, config):

    data_handler.load_test_data()

    svm_train_size = 100

    x_syn_data, label_syn_data = [], []

    z_pl = tf.placeholder(tf.float32, shape=[None, config.z_dim])
    c_pl = tf.placeholder(tf.float32, shape=[None, config.attr_dim])
    G_samples = model.G(z_pl, c_pl, reuse=True)
        
    z = sample_z(svm_train_size, config.z_dim)
    for idx, ci_attr in enumerate(data_handler.test_attr):
        xi_syn_data = sess.run([G_samples], feed_dict={z_pl:z, c_pl:np.tile(ci_attr, (svm_train_size, 1))})
        xi_syn_data=np.squeeze(xi_syn_data, axis=0)
        x_syn_data.extend(xi_syn_data)
        label_syn_data.extend(data_handler.test_class_index[idx] * np.ones((svm_train_size)))
    
    svm_model = LinearSVM(config) 
    svm_model.train(x_syn_data, label_syn_data)
    accuracy = svm_model.measure_accuracy(data_handler.test_data, data_handler.test_label)
    
    summ = sess.run(summ_op, feed_dict={acc_val:accuracy})
    return summ
        
    
def sample_z(batch_size, z_dimen):
    return np.random.normal(0., 1., size=[batch_size, z_dimen])
#return np.random.uniform(-1., 1., size=[batch_size, z_dimen])

if __name__ == '__main__':
    main()
    
