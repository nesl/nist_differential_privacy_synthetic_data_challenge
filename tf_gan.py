"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import sys
import pdb
import math
import numpy as np
import data_utils
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import time

from tensorflow.distributions import Bernoulli, Categorical

from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from differential_privacy.dp_sgd.dp_optimizer import sanitizer
from differential_privacy.dp_sgd.dp_optimizer import utils
from differential_privacy.privacy_accountant.tf import accountant


flags = tf.app.flags
flags.DEFINE_string('input_file', 'input.csv', 'Input file')
flags.DEFINE_string('output_file', 'output.csv', 'output file')
flags.DEFINE_string('meta_file', 'metadata.json', 'metadata file')
flags.DEFINE_float('epsilon', 8.0, 'Target eps')
flags.DEFINE_float('delta', None, 'maximum delta')
# Training parameters
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_integer('num_epochs', 20, 'Number of training epochs')
flags.DEFINE_integer(
    'save_every', 1, 'Save training logs every how many epochs')
flags.DEFINE_float('weight_clip', 0.01, 'weight clipping value')
# Model parameters
flags.DEFINE_integer('z_size', 64, 'Size of input size')
flags.DEFINE_integer('hidden_dim', 1024, 'Size of hidden layer')


# Privacy parameters
flags.DEFINE_bool('with_privacy', False, 'Turn on/off differential privacy')
flags.DEFINE_float('gradient_l2norm_bound', 1.0, 'l2 norm clipping')
# Sampling and model restore
flags.DEFINE_integer('sampling_size', 100000, 'Number of examples to sample')
flags.DEFINE_string('checkpoint', None, 'Checkpoint to restore')
flags.DEFINE_bool('sample', False, 'Perform sampling')
flags.DEFINE_bool('dummy', False,
                  'If True, then test our model using dummy data ')


#########################################################################
# Utility functions for building the WGAN model
#########################################################################
def lrelu(x, alpha=0.01):
    """ leaky relu activation function """
    return tf.nn.leaky_relu(x, alpha)


def fully_connected(input_node, output_dim, activation=tf.nn.relu, scope='None'):
    """ returns both the projection and output activation """
    with tf.variable_scope(scope or 'FC'):
        w = tf.get_variable('w', shape=[input_node.get_shape()[1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', shape=[output_dim],
                            initializer=tf.constant_initializer())
        tf.summary.histogram('w', w)
        tf.summary.histogram('b', b)
        z = tf.matmul(input_node, w) + b
        h = activation(z)
    return z, h


def critic_f(input_node, hidden_dim):
    """ Defines the critic model architecture """
    z1, h1 = fully_connected(input_node, hidden_dim, lrelu, scope='fc1')
    # z2, h2 = fully_connected(h1, hidden_dim, lrelu, scope='fc2')
    z3, _ = fully_connected(h1, 1, tf.identity, scope='fc3')
    return z3


def generator(input_node, hidden_dim, output_dim):
    """ Defines the generator model architecture """
    z1, h1 = fully_connected(input_node, hidden_dim, lrelu, scope='fc1')
    # z2, h2 = fully_connected(h1, hidden_dim, lrelu, scope='fc2')
    z3, _ = fully_connected(h1, output_dim, tf.identity, scope='fc3')
    return z3


def nist_data_format(output, metadata, columns_list, col_maps):
    """ Output layer format for generator data """
    with tf.name_scope('nist_format'):
        output_list = []
        cur_idx = 0
        for k in columns_list:
            v = col_maps[k]
            if isinstance(v, dict):
                if len(v) == 2:
                    output_list.append(tf.nn.sigmoid(
                        output[:, cur_idx:cur_idx+1]))
                    cur_idx += 1
                else:
                    output_list.append(
                        tf.nn.softmax(output[:, cur_idx: cur_idx+len(v)]))
                    cur_idx += len(v)
            elif v == 'int':
                output_list.append(output[:, cur_idx:cur_idx+1])
                cur_idx += 1
            elif v == 'int_v':
                output_list.append(tf.nn.sigmoid(output[:, cur_idx:cur_idx+1]))
                output_list.append(output[:, cur_idx+1:cur_idx+2])
                cur_idx += 2
            elif v == 'void':
                pass
            else:
                raise Exception('ivnalid mapping for col {}'.format(k))
        return tf.concat(output_list, axis=1)


def nist_sampling_format(output, metadata, columns_list, col_maps):
    """
    Output layer format for generator data plus performing random sampling
     from the output softmax and bernoulli distributions.
    """
    with tf.name_scope('nist_sampling_format'):
        output_list = []
        cur_idx = 0
        for k in columns_list:
            v = col_maps[k]
            if isinstance(v, dict):
                if len(v) == 2:
                    output_list.append(
                        tf.cast(
                            tf.expand_dims(
                                Bernoulli(logits=output[:, cur_idx]).sample(), axis=1), tf.float32)
                    )
                    cur_idx += 1
                else:
                    output_list.append(
                        tf.cast(tf.expand_dims(
                            Categorical(logits=output[:, cur_idx: cur_idx+len(v)]).sample(), axis=1), tf.float32))

                    cur_idx += len(v)
            elif v == 'int':
                output_list.append(
                    tf.nn.relu(output[:, cur_idx:cur_idx+1]))
                cur_idx += 1
            elif v == 'int_v':
                output_list.append(tf.nn.sigmoid(output[:, cur_idx:cur_idx+1]))
                output_list.append(tf.nn.relu(output[:, cur_idx+1:cur_idx+2]))
                cur_idx += 2
            elif v == 'void':
                pass
        return tf.concat(output_list, axis=1)


def sample_dataset(sess, sampling_output, output_fname, columns_list, sampling_size):
    """ Performs sampling to output synthetic data from the generative model.
    Saves the result to output_fname file.
    """
    sampling_result = []
    num_samples = 0
    while num_samples < sampling_size:
        batch_samples = sess.run(sampling_output)
        num_samples += batch_samples.shape[0]
        sampling_result.append(batch_samples)
    sampling_result = np.concatenate(sampling_result, axis=0)
    print(sampling_result.shape)
    final_df = data_utils.postprocess_data(
        sampling_result, metadata, col_maps, columns_list, greedy=False)
    print(final_df.shape)
    final_df = pd.DataFrame(
        data=final_df, columns=original_df.columns, index=None)
    final_df.to_csv(output_fname, index=False)


if __name__ == '__main__':
    FLAGS = flags.FLAGS

    # Reading input data
    original_df, input_data, metadata, col_maps, columns_list = data_utils.preprocess_nist_data(
        FLAGS.input_file, FLAGS.meta_file, subsample=False)

    input_data = input_data.values  # .astype(np.float32)
    data_dim = input_data.shape[1]
    format_fun = nist_data_format
    num_examples = input_data.shape[0]
    print('** Reading input ** ')
    print('-- Read {} rows, {} columns ----'.format(num_examples, data_dim))

    batch_size = FLAGS.batch_size
    print('Batch size = ', batch_size)
    num_batches = math.ceil(num_examples / batch_size)
    T = FLAGS.num_epochs * num_batches
    q = float(FLAGS.batch_size) / num_examples

    max_eps = FLAGS.epsilon

    if FLAGS.delta is None:
        max_delta = 1.0 / (num_examples**2)
    else:
        max_delta = FLAGS.delta

    print('Privacy budget = ({}, {})'.format(max_eps, max_delta))
    # Decide which accountanint_v to use
    use_moments_accountant = max_eps > 0.7
    if use_moments_accountant:
        if max_eps > 5.0:
            sigma = 1.0
        else:
            sigma = 3.0
        eps_per_step = None  # unused for moments accountant
        delta_per_step = None  # unused for moments accountant
        print('Using moments accountant (\sigma = {})'.format(sigma))
    else:
        sigma = None  # unused for amortized accountant
        # bound of eps_per_step from lemma 2.3 in https://arxiv.org/pdf/1405.7085v2.pdf
        eps_per_step = max_eps / (q * math.sqrt(2 * T * math.log(1/max_delta)))
        delta_per_step = max_delta / (T * q)
        print('Using amortized accountant (\eps, \delta)-per step = ({},{})'.format(
            eps_per_step, delta_per_step))
    with tf.name_scope('inputs'):
        x_holder = tf.placeholder(tf.float32, [None, data_dim], 'x')
        z_holder = tf.random_normal(shape=[FLAGS.batch_size, FLAGS.z_size],
                                    dtype=tf.float32, name='z')
        sampling_noise = tf.random_normal([FLAGS.batch_size, FLAGS.z_size],
                                          dtype=tf.float32, name='sample_z')
        eps_holder = tf.placeholder(tf.float32, [], 'eps')
        delta_holder = tf.placeholder(tf.float32, [], 'delta')

    print("Data Dimention: ", data_dim)
    print("X Holder: ", x_holder)
    print("Z Holder: ", z_holder)
    with tf.variable_scope('generator') as scope:
        gen_output = generator(z_holder, FLAGS.hidden_dim, data_dim)
        print(gen_output)
        gen_output = format_fun(gen_output, metadata, columns_list, col_maps)
        print(gen_output)
        scope.reuse_variables()
        sampling_output = generator(sampling_noise, FLAGS.hidden_dim, data_dim)
        sampling_output = nist_sampling_format(
            sampling_output, metadata, columns_list, col_maps)
    print(sampling_output)
    with tf.variable_scope('critic') as scope:
        critic_real = critic_f(x_holder, FLAGS.hidden_dim)
        scope.reuse_variables()
        critic_fake = critic_f(gen_output, FLAGS.hidden_dim)

    with tf.name_scope('train'):
        global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        loss_critic_real = - tf.reduce_mean(critic_real)
        loss_critic_fake = tf.reduce_mean(critic_fake)
        loss_critic = loss_critic_real + loss_critic_fake
        critic_vars = [x for x in tf.trainable_variables()
                       if x.name.startswith('critic')]
        if FLAGS.with_privacy:
            # assert FLAGS.sigma > 0, 'Sigma has to be positive when with_privacy=True'
            with tf.name_scope('privacy_accountant'):
                if use_moments_accountant:
                    # Moments accountant introduced in (https://arxiv.org/abs/1607.00133)
                    # we use same implementation of
                    # https://github.com/tensorflow/models/blob/master/research/differential_privacy/privacy_accountant/tf/accountant.py
                    priv_accountant = accountant.GaussianMomentsAccountant(
                        num_examples)
                else:
                    # AmortizedAccountant which tracks the privacy spending in the amortized way.
                    # It uses privacy amplication via sampling to compute the privacyspending for each
                    # batch and strong composition (specialized for Gaussian noise) for
                    # accumulate the privacy spending (http://arxiv.org/pdf/1405.7085v2.pdf)
                    # we use the implementation of
                    # https://github.com/tensorflow/models/blob/master/research/differential_privacy/privacy_accountant/tf/accountant.py
                    priv_accountant = accountant.AmortizedAccountant(
                        num_examples)

                # per-example Gradient l_2 norm bound.
                example_gradient_l2norm_bound = FLAGS.gradient_l2norm_bound / FLAGS.batch_size

                # Gaussian sanitizer, will enforce differential privacy by clipping the gradient-per-example.
                # Add gaussian noise, and sum the noisy gradients at each weight update step.
                # It will also notify the privacy accountant to update the privacy spending.
                gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(
                    priv_accountant,
                    [example_gradient_l2norm_bound, True])

                critic_step = dp_optimizer.DPGradientDescentOptimizer(
                    FLAGS.lr,
                    # (eps, delta) unused parameters for the moments accountant which we are using
                    [eps_holder, delta_holder],
                    gaussian_sanitizer,
                    sigma=sigma,
                    batches_per_lot=1,
                    var_list=critic_vars).minimize((loss_critic_real, loss_critic_fake),
                                                   global_step=global_step, var_list=critic_vars)

        else:
            # This is used when we train without privacy.
            critic_step = tf.train.RMSPropOptimizer(FLAGS.lr).minimize(
                loss_critic, var_list=critic_vars)

        # Weight clipping to ensure the critic function is K-Lipschitz as required
        # for WGAN training.
        clip_c = [tf.assign(var, tf.clip_by_value(
            var, -FLAGS.weight_clip, FLAGS.weight_clip)) for var in critic_vars]
        with tf.control_dependencies([critic_step]):
            critic_step = tf.tuple(clip_c)

        # Traing step of generator
        generator_vars = [x for x in tf.trainable_variables()
                          if x.name.startswith('generator')]
        loss_generator = -tf.reduce_mean(critic_fake)
        generator_step = tf.train.RMSPropOptimizer(FLAGS.lr).minimize(
            loss_generator, var_list=generator_vars)

        weight_summaries = tf.summary.merge_all()
        tb_c_op = tf.summary.scalar('critic_loss', loss_critic)
        tb_g_op = tf.summary.scalar('generator_loss', loss_generator)

    final_eps = 0.0
    final_delta = 0.0
    critic_iters = 10
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('./logs', sess.graph)
        summary_writer.flush()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if FLAGS.checkpoint:
            #  Load the model
            saver.restore(sess, FLAGS.checkpoint)
            if FLAGS.sample:
                sample_dataset(sess, sampling_output,
                               FLAGS.output_file, columns_list, FLAGS.sampling_size)
                assert FLAGS.checkpoint is not None, "You must provide a checkpoint."
                sys.exit(0)

        abort_early = False  # Flag that will be changed to True if we exceed the privacy budget
        for e in range(FLAGS.num_epochs):
            if abort_early:
                break
            # One epoch is one full pass over the whole training data
            start_time = time.time()
            # Randomly shuffle the data at the beginning of each epoch
            rand_idxs = np.arange(num_examples)
            np.random.shuffle(rand_idxs)
            idx = 0
            abort_early = False
            while idx < num_batches and not abort_early:
                if idx % 10 == 0:
                    sys.stdout.write('\r{}/{}'.format(idx, num_batches))
                    sys.stdout.flush()
                critic_i = 0
                while critic_i < critic_iters and idx < num_batches and not abort_early:
                    # Train the critic.
                    batch_idxs = rand_idxs[idx*batch_size: (idx+1)*batch_size]
                    batch_xs = input_data[batch_idxs, :]
                    feed_dict = {x_holder: batch_xs,
                                 eps_holder: eps_per_step,
                                 delta_holder: delta_per_step
                                 }
                    _, tb_c = sess.run(
                        [critic_step, tb_c_op], feed_dict=feed_dict)
                    critic_i += 1
                    idx += 1

                    if FLAGS.with_privacy:
                        if use_moments_accountant:
                            spent_eps_deltas = priv_accountant.get_privacy_spent(
                                sess, target_deltas=[max_delta])[0]
                        else:
                            spent_eps_deltas = priv_accountant.get_privacy_spent(
                                sess, target_eps=None)[0]

                        # Check whether we exceed the privacy budget
                        if (spent_eps_deltas.spent_delta > max_delta or
                                spent_eps_deltas.spent_eps > max_eps):
                            abort_early = True
                            print(
                                "\n*** Discriminator training exceeded privacy budget, aborting the training of generator ****")
                        else:
                            final_eps = spent_eps_deltas.spent_eps
                            final_delta = spent_eps_deltas.spent_delta
                    else:
                        # Training without privacy
                        spent_eps_deltas = accountant.EpsDelta(np.inf, 1)

                # Train the generator
                if not abort_early:
                    # Check for abort_early because we stop updating the generator
                    #  once we exceeded privacy budget.
                    privacy_summary = summary_pb2.Summary(value=[
                        summary_pb2.Summary.Value(tag='eps',
                                                  simple_value=final_eps)])
                    summary_writer.add_summary(privacy_summary, e)
                    _, tb_g = sess.run([generator_step, tb_g_op])
                    if e % FLAGS.save_every == 0 or (e == FLAGS.num_epochs-1):
                        summary_writer.add_summary(tb_g, e)
                end_time = time.time()

            if (e % FLAGS.save_every == 0) or (e == FLAGS.num_epochs-1) or abort_early:
                summary_writer.add_summary(tb_c, e)
                weight_summary_out = sess.run(
                    weight_summaries, feed_dict=feed_dict)
                summary_writer.add_summary(weight_summary_out, e)
                print('\nEpoch {} took {} seconds. Privacy = ({}, {}).'.format(
                    e, (end_time-start_time), spent_eps_deltas.spent_eps, spent_eps_deltas.spent_delta))
                summary_writer.flush()

        if FLAGS.with_privacy:
            print('\nTotal (\eps, \delta) privacy loss spent in training = ({}, {})'.format(
                final_eps, final_delta))
        summary_writer.close()
        # Sample synthetic data from the model after training is done.
        sample_dataset(sess, sampling_output,
                       FLAGS.output_file, columns_list, FLAGS.sampling_size)
