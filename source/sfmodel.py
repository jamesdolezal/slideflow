# Copyright (C) James Dolezal - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

# Update 3/2/2019: Beginning tf.data implementation
# Update 5/29/2019: Supports both loose image tiles and TFRecords, 
#   annotations supplied by separate annotation file upon initial model call

''''Builds a CNN model.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys
import shutil
from datetime import datetime

import numpy as np
import pickle
import argparse
import gc
import csv
import random

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorflow.python.framework import ops

from glob import glob
from scipy.stats import linregress
from statistics import median
from numpy.random import choice
from sklearn import metrics
from matplotlib import pyplot as plt

from util import tfrecords, sfutil
from util.sfutil import TCGAAnnotations, log

BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_CASE = 'BALANCE_BY_CASE'
NO_BALANCE = 'NO_BALANCE'

class HyperParameters:
	_OptDict = {
		'Adam':	tf.keras.optimizers.Adam,
		'SGD': tf.keras.optimizers.SGD,
		'RMSprop': tf.keras.optimizers.RMSprop,
		'Adagrad': tf.keras.optimizers.Adagrad,
		'Adadelta': tf.keras.optimizers.Adadelta,
		'Adamax': tf.keras.optimizers.Adamax,
		'Nadam': tf.keras.optimizers.Nadam
	}
	_ModelDict = {
		'Xception': tf.keras.applications.Xception,
		'VGG16': tf.keras.applications.VGG16,
		'VGG19': tf.keras.applications.VGG19,
		'ResNet50': tf.keras.applications.ResNet50,
		#'ResNet101': tf.keras.applications.ResNet101,
		#'ResNet152': tf.keras.applications.ResNet152,
		#'ResNet50V2': tf.keras.applications.ResNet50V2,
		#'ResNet101V2': tf.keras.applications.ResNet101V2,
		#'ResNet152V2': tf.keras.applications.ResNet152V2,
		#'ResNeXt50': tf.keras.applications.ResNeXt50,
		#'ResNeXt101': tf.keras.applications.ResNeXt101,
		'InceptionV3': tf.keras.applications.InceptionV3,
		'InceptionResNetV2': tf.keras.applications.InceptionResNetV2,
		'MobileNet': tf.keras.applications.MobileNet,
		'MobileNetV2': tf.keras.applications.MobileNetV2,
		#'DenseNet': tf.keras.applications.DenseNet,
		#'NASNet': tf.keras.applications.NASNet
	}
	def __init__(self, toplayer_epochs=0, finetune_epochs=50, model='InceptionV3', pooling='avg', loss='sparse_categorical_crossentropy',
				 learning_rate=0.1, batch_size=16, hidden_layers=0, optimizer='Adam', early_stop=False, 
				 early_stop_patience=0, balanced_training=BALANCE_BY_CATEGORY, balanced_validation=NO_BALANCE, 
				 augment=True):
		''' Additional hyperparameters to consider:
		beta1 0.9
		beta2 0.999
		epsilon 1.0
		batch_norm_decay 0.99
		'''
		self.toplayer_epochs = toplayer_epochs
		self.finetune_epochs = finetune_epochs
		self.model = model
		self.pooling = pooling
		self.loss = loss
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.optimizer = optimizer
		self.early_stop = early_stop
		self.hidden_layers = hidden_layers
		self.early_stop_patience = early_stop_patience
		self.balanced_training = balanced_training
		self.balanced_validation = balanced_validation
		self.augment = augment

	def get_opt(self):
		return self._OptDict[self.optimizer](lr=self.learning_rate)

	def get_model(self, input_shape, weights):
		return self._ModelDict[self.model](
			input_shape=input_shape,
			include_top=False,
			pooling=self.pooling,
			weights=weights
		)

	def _get_args(self):
		return [arg for arg in dir(self) if not arg[0]=='_' and arg not in ['get_opt', 'get_model']]

	def __str__(self):
		output = "Hyperparameters:\n"
			
		args = self._get_args()
		for arg in args:
			value = getattr(self, arg)
			output += log.empty(f"{sfutil.header(arg)} = {value}\n", 2, None)
		return output

class SlideflowModel:
	''' Model containing all functions necessary to build input dataset pipelines,
	build a training and validation set model, and monitor and execute training.'''
	def __init__(self, data_directory, input_directory, image_size, slide_to_category, validation_strategy='per-tile', validation_fraction=None, manifest=None, use_fp16=True):
		self.DATA_DIR = data_directory # Directory where to write event logs and checkpoints.
		self.INPUT_DIR = input_directory
		self.MANIFEST = manifest
		self.IMAGE_SIZE = image_size
		self.USE_FP16 = use_fp16
		self.DTYPE = tf.float16 if self.USE_FP16 else tf.float32
		self.SLIDES = list(slide_to_category.keys()) # If None, will default to using all tfrecords in the input directory
		self.SLIDE_TO_CATEGORY = slide_to_category # Dictionary mapping slide names to category
		if validation_strategy=='per-slide':
			self.VALIDATION_SLIDES = random.sample(self.SLIDES, int(validation_fraction * len(self.SLIDES)))
		else:
			self.VALIDATION_SLIDES = None
		self.NUM_CLASSES = len(list(set(slide_to_category.values())))

		with tf.device('/cpu'):
			self.ANNOTATIONS_TABLE = tf.lookup.StaticHashTable(
				tf.lookup.KeyValueTensorInitializer(list(slide_to_category.keys()), list(slide_to_category.values())), -1
			)

		if not os.path.exists(self.DATA_DIR):
			os.makedirs(self.DATA_DIR)

	def _process_image(self, image_string, augment):
		image = tf.image.decode_jpeg(image_string, channels = 3)
		image = tf.image.per_image_standardization(image)

		if augment:
			# Apply augmentations
			# Rotate 0, 90, 180, 270 degrees
			image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

			# Random flip and rotation
			image = tf.image.random_flip_left_right(image)
			image = tf.image.random_flip_up_down(image)

		dtype = tf.float16 if self.USE_FP16 else tf.float32
		image = tf.image.convert_image_dtype(image, dtype)
		image.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
		return image

	def _parse_tfrecord_function(self, record):
		features = tf.io.parse_single_example(record, tfrecords.FEATURE_DESCRIPTION)
		case = features['case']
		label = self.ANNOTATIONS_TABLE.lookup(case)
		image_string = features['image_raw']
		image = self._process_image(image_string, self.AUGMENT)
		return image, label

	def _interleave_tfrecords(self, folder, batch_size, balance, finite, dataset=None):
		'''Generates an interleaved dataset from a collection of tfrecord files,
		sampling from tfrecord files randomly according to balancing if provided.
		Requires self.MANIFEST. Assumes TFRecord files are named by case.

		Args:
			folder		Location to search for TFRecord files
			batch_size	Batch size
			balance		Whether to use balancing for batches. Options are BALANCE_BY_CATEGORY,
							BALANCE_BY_CASE, and NO_BALANCE. If finite option is used, will drop
							tiles in order to maintain proportions across the interleaved dataset.
			augment		Whether to use data augmentation (random flip/rotate)
			finite		Whether create finite or infinite datasets. WARNING: If finite option is 
							used with balancing, some tiles will be skipped.'''
		datasets = []
		datasets_categories = []
		num_tiles = []
		global_num_tiles = 0
		categories = {}
		categories_prob = {}
		categories_tile_fraction = {}
		search_folder = os.path.join(self.INPUT_DIR, folder)
		tfrecord_files = glob(os.path.join(search_folder, "*.tfrecords"))
		if tfrecord_files == []:
			log.error(f"No TFRecords found in {sfutil.green(search_folder)}", 1)
			sys.exit()
		if not dataset:
			# Now remove all tfrecord_files except those in self.SLIDES (if not None)
			tfrecord_files = tfrecord_files if not self.SLIDES else [tfr for tfr in tfrecord_files 
																	if tfr.split('/')[-1][:-10] in self.SLIDES]
		elif dataset=='train':
			tfrecord_files = tfrecord_files if not self.SLIDES else [tfr for tfr in tfrecord_files 
																	if (tfr.split('/')[-1][:-10] in self.SLIDES) and (tfr.split('/')[-1][:-10] not in self.VALIDATION_SLIDES)]
			log.info(f"Using {sfutil.green(str(len(tfrecord_files)))} files for training set", 1)
		elif dataset=='validation':
			tfrecord_files = tfrecord_files if not self.SLIDES else [tfr for tfr in tfrecord_files 
																	if (tfr.split('/')[-1][:-10] in self.SLIDES) and (tfr.split('/')[-1][:-10] in self.VALIDATION_SLIDES)]
			log.info(f"Using {sfutil.green(str(len(tfrecord_files)))} files for validation set", 1)
		for filename in tfrecord_files:
			dataset_to_add = tf.data.TFRecordDataset(filename) if finite else tf.data.TFRecordDataset(filename).repeat()
			datasets += [dataset_to_add]
			slide_name = filename.split('/')[-1][:-10]
			category = self.SLIDE_TO_CATEGORY[slide_name]
			datasets_categories += [category]
			tiles = self.MANIFEST[filename]['total']
			if category not in categories.keys():
				categories.update({category: {'num_cases': 1,
											  'num_tiles': tiles}})
			else:
				categories[category]['num_cases'] += 1
				categories[category]['num_tiles'] += tiles
			num_tiles += [tiles]
		for category in categories:
			lowest_category_case_count = min([categories[i]['num_cases'] for i in categories])
			lowest_category_tile_count = min([categories[i]['num_tiles'] for i in categories])
			categories_prob[category] = lowest_category_case_count / categories[category]['num_cases']
			categories_tile_fraction[category] = lowest_category_tile_count / categories[category]['num_tiles']
		if balance == NO_BALANCE:
			log.info(f"Not balancing input from {sfutil.green(folder)}", 1)
			prob_weights = [i/sum(num_tiles) for i in num_tiles]
		if balance == BALANCE_BY_CASE:
			log.info(f"Balancing input from {sfutil.green(folder)} across cases", 1)
			prob_weights = None
			if finite:
				# Only take as many tiles as the number of tiles in the smallest dataset
				for i in range(len(datasets)):
					num_to_take = min(num_tiles)
					datasets[i] = datasets[i].take(num_to_take)
					global_num_tiles += num_to_take
		if balance == BALANCE_BY_CATEGORY:
			log.info(f"Balancing input from {sfutil.green(folder)} across categories", 1)
			prob_weights = [categories_prob[datasets_categories[i]] for i in range(len(datasets))]
			if finite:
				# Only take as many tiles as the number of tiles in the smallest category
				for i in range(len(datasets)):
					num_to_take = num_tiles[i] * categories_tile_fraction[datasets_categories[i]]
					datasets[i] = datasets[i].take(num_to_take)
					global_num_tiles += num_to_take
		# Remove empty cases
		for i in sorted(range(len(prob_weights)), reverse=True):
			if num_tiles[i] == 0:
				del(num_tiles[i])
				del(datasets[i])
				del(datasets_categories[i])
				del(prob_weights[i])
		# If the global tile count was not manually set as above, then assume it is the sum of all tiles across all slides
		if global_num_tiles==0:
			global_num_tiles = sum(num_tiles)
		try:
			dataset = tf.data.experimental.sample_from_datasets(datasets, weights=prob_weights)
		except IndexError:
			log.error(f"No TFRecords found in {sfutil.green(search_folder)} after filter criteria", 1)
			sys.exit()
		dataset = dataset.map(self._parse_tfrecord_function, num_parallel_calls = 8)
		dataset = dataset.batch(batch_size)
		return dataset, global_num_tiles

	def build_dataset_inputs(self, subfolder, batch_size, balance, augment, finite=False, dataset=None):
		'''Args:
			subfolder:		Sub-directory in which to search for tfrecords, if applicable
			balance:		Whether to use input balancing; options are BALANCE_BY_CASE, BALANCE_BY_CATEGORY, NO_BALANCE
								 (only available if TFRECORDS_BY_CASE=True)'''
		self.AUGMENT = augment
		with tf.name_scope('input'):
			dataset, num_tiles = self._interleave_tfrecords(subfolder, batch_size, balance, finite, dataset)
		return dataset, num_tiles

	def build_model(self, hp, pretrain=None, checkpoint=None):
		# Assemble base model, using pretraining (imagenet) or the base layers of a supplied model
		if pretrain:
			log.info(f"Using pretraining from {sfutil.green(pretrain)}", 1)
		if pretrain and pretrain!='imagenet':
			# Load pretrained model
			pretrained_model = tf.keras.models.load_model(pretrain)
			base_model = pretrained_model.get_layer(index=0)
		else:
			# Create model using ImageNet if specified
			base_model = hp.get_model(input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
									  weights=pretrain)

		# Combine base model with top layer (classification/prediction layer)
		layers = [base_model]
		if not hp.pooling:
			layers += [tf.keras.layers.Flatten()]
		# Add hidden layers if specified
		for i in range(hp.hidden_layers):
			layers += [tf.keras.layers.Dense(500, activation='relu')]
		# If no hidden layers and no pooling is used, flatten the output prior to softmax
		
		# Add the softmax prediction layer
		layers += [tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax')]
		model = tf.keras.Sequential(layers)
		
		if checkpoint:
			log.info(f"Loading checkpoint weights from {sfutil.green(checkpoint)}", 1)
			model.load_weights(checkpoint)

		return model

	def generate_roc(self, y_true, y_pred, name='ROC'):
		fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
		roc_auc = metrics.auc(fpr, tpr)

		# Plot
		plt.clf()
		plt.title('ROC Curve')
		plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('TPR')
		plt.xlabel('FPR')
		plt.savefig(os.path.join(self.DATA_DIR, f'{name}.png'))

	def generate_predictions_and_roc(self, model, dataset):
		# Get predictions and performance metrics
		log.info("Generating predictions...", 1)
		y_true, y_pred = [], []
		for batch in dataset:
			y_true += [batch[1].numpy()]
			y_pred += [model.predict_on_batch(batch)]
		y_pred = np.concatenate(y_pred)

		# Convert y_true to one_hot encoding
		num_cat = len(y_pred[0])
		def to_onehot(val):
			onehot = [0] * num_cat
			onehot[val] = 1
			return onehot
		y_true = np.array([to_onehot(i) for i in np.concatenate(y_true)])

		# Generate ROC
		for i in range(num_cat):
			self.generate_roc(y_true[:, i], y_pred[:, i], f'ROC{i}')

		# Save results to CSV
		csv_dir = os.path.join(self.DATA_DIR, "eval_predictions.csv")
		with open(csv_dir, 'w') as outfile:
			writer = csv.writer(outfile)
			header = [f"y_true{i}" for i in range(num_cat)] + [f"y_pred{j}" for j in range(num_cat)]
			writer.writerow(header)
			for i in range(len(y_true)):
				row = np.concatenate([y_true[i], y_pred[i]])
				writer.writerow(row)
		log.complete(f"Predictions saved to {sfutil.green(csv_dir)}", 1)


	def evaluate(self, subdir="validation", hp=None, model=None, checkpoint=None, batch_size=None):
		# Load and initialize model
		if not hp and checkpoint:
			log.error("If using a checkpoint for evaluation, hyperparameters must be specified.")
			sys.exit()
		batch_size = batch_size if not hp else hp.batch_size
		augment = False if not hp else hp.augment
		data_to_eval, _ = self.build_dataset_inputs(subdir, batch_size, NO_BALANCE, augment, finite=True)
		if model:
			self.model = tf.keras.models.load_model(model)
		elif checkpoint:
			self.model = self.build_model(hp)
			self.model.load_weights(checkpoint)

		self.generate_predictions_and_roc(self.model, data_to_eval)

		log.info("Calculating performance metrics...", 1)
		results = self.model.evaluate(data_to_eval)

		return results

	def retrain_top_layers(self, model, hp, train_data, validation_data, steps_per_epoch, callbacks=None, epochs=1, verbose=1):
		if verbose: log.info("Retraining top layer", 1)
		# Freeze the base layer
		model.layers[0].trainable = False
		val_steps = 100 if validation_data else None

		model.compile(optimizer=tf.keras.optimizers.Adam(lr=hp.learning_rate),
					  loss=hp.loss,
					  metrics=['accuracy'])

		toplayer_model = model.fit(train_data,
				  epochs=epochs,
				  verbose=verbose,
				  steps_per_epoch=steps_per_epoch,
				  validation_data=validation_data,
				  validation_steps=val_steps,
				  callbacks=callbacks)

		# Unfreeze the base layer
		model.layers[0].trainable = True
		return toplayer_model.history

	def train(self, hp, pretrain='imagenet', resume_training=None, checkpoint=None, supervised=True, log_frequency=20):
		'''Train the model for a number of steps, according to flags set by the argument parser.'''

		# Build inputs
		validation_subfolder = 'train' if self.VALIDATION_SLIDES else 'validation'
		train_data, num_tiles = self.build_dataset_inputs('train', hp.batch_size, hp.balanced_training, hp.augment, dataset='train')
		validation_data, _ = self.build_dataset_inputs(validation_subfolder, hp.batch_size, hp.balanced_validation, hp.augment, finite=supervised, dataset='validation')
		#training_val_data = validation_data.repeat() if supervised else None
		
		#testing overide
		#num_tiles = 100
		#hp.finetune_epochs = 1

		# Calculate parameters
		total_epochs = hp.toplayer_epochs + hp.finetune_epochs
		initialized_optimizer = hp.get_opt()
		steps_per_epoch = round(num_tiles/hp.batch_size)
		tf.keras.layers.BatchNormalization = sfutil.UpdatedBatchNormalization
		verbose = 1 if supervised else 0
		val_steps = 200# if supervised else None

		# Create callbacks for early stopping, checkpoint saving, summaries, and history
		history_callback = tf.keras.callbacks.History()
		early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=hp.early_stop_patience)
		checkpoint_path = os.path.join(self.DATA_DIR, "cp.ckpt")
		cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
														save_weights_only=True,
														verbose=1)
		
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.DATA_DIR, 
															histogram_freq=0,
															write_graph=False,
															update_freq=hp.batch_size*log_frequency)
		callbacks = [history_callback]
		if hp.early_stop:
			callbacks += [early_stop_callback]
		if supervised:
			callbacks += [cp_callback, tensorboard_callback]

		# Build or load model
		if resume_training:
			if verbose:	log.info(f"Resuming training from {sfutil.green(resume_training)}", 1)
			self.model = tf.keras.models.load_model(resume_training)
		else:
			self.model = self.build_model(hp, pretrain=pretrain, checkpoint=checkpoint)

		# Retrain top layer only if using transfer learning and not resuming training
		if hp.toplayer_epochs:
			self.retrain_top_layers(self.model, hp, train_data.repeat(), training_val_data, steps_per_epoch, 
									callbacks=None, epochs=hp.toplayer_epochs, verbose=verbose)

		# Fine-tune the model
		if verbose:	log.info("Beginning fine-tuning", 1)

		self.model.compile(loss=hp.loss,
					optimizer=initialized_optimizer,
					metrics=['accuracy'])

		finetune_model = self.model.fit(train_data.repeat(),
			steps_per_epoch=steps_per_epoch,
			epochs=total_epochs,
			verbose=verbose,
			initial_epoch=hp.toplayer_epochs,
			validation_data=validation_data.repeat(),
			validation_steps=val_steps,
			callbacks=callbacks)

		self.model.save(os.path.join(self.DATA_DIR, "trained_model.h5"))
		train_acc = finetune_model.history['accuracy']

		# Generate predictions and ROC
		self.generate_predictions_and_roc(self.model, validation_data)

		# Final validation testing, getting both overall accuracy/loss and predictions for ROCs
		if verbose: log.info("Beginning validation testing", 1)
		val_loss, val_acc = self.model.evaluate(validation_data, verbose=0)

		return train_acc, val_loss, val_acc