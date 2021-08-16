import tensorflow as tf
import os
import tempfile
import numpy as np
from slideflow.util import log

class HyperParameterError(Exception):
	pass

class ManifestError(Exception):
	pass

class ModelError(Exception):
	def __init__(self, message, errors=None):
		log.error(message)
		super().__init__(message)

class no_scope():
	def __enter__(self):
		return None
	def __exit__(self, exc_type, exc_value, traceback):
		return False

def get_layer_index_by_name(model, name):
	for i, layer in enumerate(model.layers):
			if layer.name == name:
				return i

def _negative_log_likelihood(y_true, y_pred):
	'''
	First implementation, mostly by Fred.
	Looks like it was adapted from here: https://github.com/havakv/pycox/blob/master/pycox/models/loss.py'''
	events = tf.reshape(y_pred[:, -1], [-1]) # E
	pred_hr = tf.reshape(y_pred[:, 0], [-1]) # y_pred
	time = tf.reshape(y_true, [-1])		   # y_true

	order = tf.argsort(time) #direction='DESCENDING'
	sorted_events = tf.gather(events, order) 		# E
	sorted_predictions = tf.gather(pred_hr, order) 	# y_pred

	# Finds maximum HR in predictions
	gamma = tf.math.reduce_max(sorted_predictions)

	# Small constant value
	eps = tf.constant(1e-7, dtype=tf.float32)

	log_cumsum_h = tf.math.add(
					tf.math.log(
						tf.math.add(
							tf.math.cumsum(
								tf.math.exp(
									tf.math.subtract(sorted_predictions, gamma))),
							eps)),
					gamma)

	neg_likelihood = -tf.math.divide(
						tf.reduce_sum(
							tf.math.multiply(
								tf.subtract(sorted_predictions, log_cumsum_h),
								sorted_events)),
						tf.reduce_sum(sorted_events))
						
	return neg_likelihood

def negative_log_likelihood(y_true, y_pred):
	'''Third attempt - Breslow approximation'''
	events = tf.reshape(y_pred[:, -1], [-1])
	pred = tf.reshape(y_pred[:, 0], [-1])
	time = tf.reshape(y_true, [-1])

	order = tf.argsort(time, direction='DESCENDING')
	sorted_time = tf.gather(time, order)
	sorted_events = tf.gather(events, order)
	sorted_pred = tf.gather(pred, order)

	Y_hat_c = sorted_pred
	Y_label_T = sorted_time
	Y_label_E = sorted_events
	Obs = tf.reduce_sum(Y_label_E)

	# numerical stability
	amax = tf.reduce_max(Y_hat_c)
	Y_hat_c_shift = tf.subtract(Y_hat_c, amax)
	#Y_hat_c_shift = tf.debugging.check_numerics(Y_hat_c_shift, message="checking y_hat_c_shift")
	Y_hat_hr = tf.exp(Y_hat_c_shift)
	Y_hat_cumsum = tf.math.log(tf.cumsum(Y_hat_hr)) + amax

	unique_values, segment_ids = tf.unique(Y_label_T)
	loss_s2_v = tf.math.segment_max(Y_hat_cumsum, segment_ids)
	loss_s2_count = tf.math.segment_sum(Y_label_E, segment_ids)

	loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v, loss_s2_count))
	loss_s1 = tf.reduce_sum(tf.multiply(Y_hat_c, Y_label_E))
	loss_breslow = tf.divide(tf.subtract(loss_s2, loss_s1), Obs)

	return loss_breslow

def _new_negative_log_likelihood(y_true, y_pred):
	'''Second attempt at implementation, by James.'''
	events = tf.reshape(y_pred[:, -1], [-1])
	pred_hr = tf.reshape(y_pred[:, 0], [-1])
	time = tf.reshape(y_true, [-1])

	order = tf.argsort(time)
	sorted_events = tf.gather(events, order)
	sorted_hr = tf.math.log(tf.gather(pred_hr, order))

	neg_likelihood = - tf.reduce_sum(
						tf.math.divide(
							tf.math.multiply(
								sorted_hr,
								sorted_events),
							tf.math.cumsum(sorted_hr, reverse=True)))
	
	return neg_likelihood

'''def _n_l_l(y_true, y_pred):
	E = y_pred[:, -1]
	y_pred = y_pred[:, :-1]
	E = tf.reshape(E, [-1])
	y_pred = tf.reshape(y_pred, [-1])
	y_true = tf.reshape(y_true, [-1])
	order = tf.argsort(y_true)
	E = tf.gather(E, order)
	y_pred = tf.gather(y_pred, order)
	gamma = tf.math.reduce_max(y_pred)
	eps = tf.constant(1e-7, dtype=tf.float16)
	log_cumsum_h = tf.math.add(tf.math.log(tf.math.add(tf.math.cumsum(tf.math.exp(tf.math.subtract(y_pred, gamma))), eps)), gamma)
	neg_likelihood = -tf.math.divide(tf.reduce_sum(tf.math.multiply(tf.subtract(y_pred, log_cumsum_h), E)),tf.reduce_sum(E))
	return neg_likelihood'''

def concordance_index(y_true, y_pred):
	E = y_pred[:, -1]
	y_pred = y_pred[:, :-1]
	E = tf.reshape(E, [-1])
	y_pred = tf.reshape(y_pred, [-1])
	y_pred = -y_pred #negative of log hazard ratio to have correct relationship with survival
	g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
	g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)
	f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
	event = tf.multiply(tf.transpose(E), E)
	f = tf.multiply(tf.cast(f, tf.float32), event)
	f = tf.compat.v1.matrix_band_part(tf.cast(f, tf.float32), -1, 0)
	g = tf.reduce_sum(tf.multiply(g, f))
	f = tf.reduce_sum(f)
	return tf.where(tf.equal(f, 0), 0.0, g/f)

def add_regularization(model, regularizer):
	'''Adds regularization (e.g. L2) to all eligible layers of a model.
	This function is from "https://sthalles.github.io/keras-regularizer/" '''
	if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
		print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
		return model

	for layer in model.layers:
		for attr in ['kernel_regularizer']:
			if hasattr(layer, attr):
				setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
	model_json = model.to_json()

	# Save the weights before reloading the model.
	tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
	model.save_weights(tmp_weights_path)

	# load the model from the config
	model = tf.keras.models.model_from_json(model_json)

	# Reload the model weights
	model.load_weights(tmp_weights_path, by_name=True)
	return model

# ====== Scratch code for new CPH models

def make_riskset(time):
	'''Compute mask that represents a sample's risk set.
	
	Args:
		time:		np array, shape=(n_samples,). Observed event time ?in descending order?.

	Returns:
		risk_set:	np array, shape=(n_samples, n_samples). Boolean matrix where the `i`-th row
						denotes the risk set of the `i`-th instance, i.e. the indices `j` 
						for which the observer time `y_j >= y_i`
	'''
	o = np.argsort(-time, kind='mergesort')
	n_samples = len(time)

	risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
	for i_org, i_sort in enumerate(o):
		k = i_org
		while k < n_samples and time[i_sort] == time[o[k]]:
			k += 1
		risk_set[i_sort, o[:k]] = True
	return risk_set