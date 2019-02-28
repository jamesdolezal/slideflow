# Copyright (C) James Dolezal - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

''''Builds the HISTCON network.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib, xrange
import tensorflow as tf

import histcon_input

parser = histcon_input.parser
FLAGS = histcon_input.FLAGS

# Global constants describing the HISTCON data set.
IMAGE_SIZE = histcon_input.IMAGE_SIZE
NUM_CLASSES = histcon_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = histcon_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = histcon_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants for the training process.
MOVING_AVERAGE_DECAY = 0.9999 		# Decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 240.0		# Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.05	# Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001			# Initial learning rate.

def _activation_summary(x):
	'''Helper to create summaries for activations.

	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.

	Args:
		x: Tensor

	Returns:
		None
	'''
	tf.summary.histogram(x.op.name + '/activations', x)
	tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
	'''Helper to create a Variable stored on CPU memory.

	Args:
		name: variable name
		shape: list of ints
		initializer: Variable initializer

	Returns:
		Variable Tensor
	'''
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var

def _variable_with_weight_decay(name, shape, stddev, wd):
	'''Helper to create an initialized Variable with weight decay.

	The Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
		name: Variable name
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
			decay is not added for this Variable.

	Returns:
		Variable Tensor
	'''
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(
		name,
		shape,
		tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def processed_inputs():
	'''Construct processed input for HISTCON training.

	Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		labels: Labels. 1D tensor of [batch_size] size.

	Raises:
		ValueError: if no data_dir
	'''
	if not FLAGS.data_dir:
		raise ValueError('Please designate a data_dir.')
	images, labels = histcon_input.processed_inputs(data_dir=FLAGS.data_dir,
													batch_size=FLAGS.batch_size,
													eval_data=False)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels

def inputs(eval_data):
	'''Construct input for HISTCON evaluation.

	Args:
		eval_data: bool, indicating if one should use the train or eval data set.

	Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		labels: Labels. 1D tensor of [batch_size] size.

	Raises:
		ValueError: if no data_dir
	'''
	if not FLAGS.data_dir:
		raise ValueError("Please designate a data_dir.")
	images, labels = histcon_input.inputs(  data_dir=FLAGS.data_dir,
											batch_size=FLAGS.batch_size,
											eval_data=eval_data)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels

def _conv_layer(input_tensor, _id, shape, ksize=None, strides=None):
	'''Helper to create convolutional layers with or without pooling.

	Args:
		input_tensor: Input Tensor.
		_id: Layer number.
		shape: Convolutional layer shape (4D).
		kisze: Size of pooling mask. If None, there is no pooling.
		strides: Stride size for pooling. If None, there is no pooling.

	Returns:
		Pooling Tensor if pooling, otherwise convolutional Tensor.
	'''
	# Convolutional layer
	with tf.variable_scope('conv%s' % _id) as scope:
		kernel = _variable_with_weight_decay('weights',
											shape=shape,
											stddev=5e-2,
											wd=0.0)
		conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [shape[-1]], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv_output = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv_output)

	# Pooling layer
	if ksize and strides:
		return tf.nn.max_pool(conv_output, ksize=ksize, strides=strides,
							padding='SAME', name='pool%s' % _id)
	else: return conv_output

def _fully_con_layer(input_tensor, _id, shape):
	'''Helper to create fully-connected layers.

	Args:
		input_tensor: Input Tensor.
		_id: Layer number.
		shape: Fully-connected layer shape (4D).

	Returns:
		Tensor.
	'''
	with tf.variable_scope('local%s' % _id) as scope:
		weights = _variable_with_weight_decay('weights', shape=shape,
												stddev=0.04, wd = 0.004)
		biases = _variable_on_cpu('biases', [shape[-1]], tf.constant_initializer(0.1))
		local = tf.nn.relu(tf.matmul(input_tensor, weights) + biases, name=scope.name)
		_activation_summary(local)

	return local

def inference(input_tensor):
	'''Build the HISTCON model.

	Layer types:
		conv: Convolutional layers with a kernel size and feature map size.
		pool: Pooling layers.
		norm: Normalization layers.
		local: Fully-connected layers.
		linear: final layer with linear transformation to produce logits.

	Args:
		input_tensor: Images Tensor returned from distorted_inputs() or inputs().

	Returns:
		Logits.
	'''

	with tf.name_scope("Convolutional_Network"):
		# Create conv + pool for layer 1
		pool1 = _conv_layer(input_tensor, 1, [5, 5, 3, 32], [1, 2, 2, 1],
											[1, 2, 2, 1])

		# Create conv + pool for layer 2
		pool2 = _conv_layer(pool1, 2, [5, 5, 32, 32], [1, 2, 2, 1],
											[1, 2, 2, 1])

		# Create conv for layer 3
		pool3 = _conv_layer(pool2, 3, [3, 3, 32, 64], None, None)

		# Create conv for layer 4
		conv4 = _conv_layer(pool3, 4, [3, 3, 64, 64], None, None)

		with tf.variable_scope('reshape_fully_connect') as scope:
			# Move output from last layer into depth so a single matrix multiplication can be performed.
			reshape = tf.reshape(conv4, [FLAGS.batch_size, -1])
			dim = reshape.get_shape()[1].value

		# local5
		local5 = _fully_con_layer(reshape, 5, [dim, 2048])

		# local6
		local6 = _fully_con_layer(local5, 6, [2048, 1024])

		# linear layer (Wx + b)
		with tf.variable_scope('softmax_linear') as scope:
			weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES],
													stddev=1/1024.0, wd=0.0)
			biases = _variable_on_cpu('biases', [NUM_CLASSES],
										tf.constant_initializer(0.0))
			softmax_linear = tf.add(tf.matmul(local6, weights), biases, name=scope.name)
			_activation_summary(softmax_linear)

	return softmax_linear

def loss(logits, labels):
	'''Add L2Loss to all trainable variables, and a summary for "Loss" and "Loss/avg".

	Args:
		logits: Logits from inference().
		labels: Labels from distorted_inputs() or inputs(). 1D tensor of shape [batch_size]

	Returns:
		Loss Tensor of type float.
	'''
	# Calculate average cross entropy loss across the batch.
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	# Total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
	'''Add summaries for losses in HISTCON model.

	Generates moving average for all losses and associated sumaries for
	visualizing network performance with Tensorboard.

	Args:
		total_loss: Total loss from loss().

	Returns:
		loss_averages_op: op fro generating moving averages of losses.
	'''
	# Compute moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss as
		# the original loss name.
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))

	return loss_averages_op

def train(total_loss, global_step):
	'''Train the HISTCON model.

	Create an optimizer and apply to all trainable variables. Add moving
	average for all trainable variables.

	Args:
		total_loss: Total loss from loss().
		global_step: Integer variable counting the number of training steps processed.

	Returns:
		train_op: op for training.
	'''

	# Variables that affect learning rate.
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	# Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
									global_step,
									decay_steps,
									LEARNING_RATE_DECAY_FACTOR,
									staircase=True)
	tf.summary.scalar('learning_rate', lr)

	# Generate moving averages of all losses and associated summaries.
	loss_averages_op = _add_loss_summaries(total_loss)

	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	# Add histograms for gradients.
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients', grad)

	# Track moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op
