import os
import io
import sys
import shutil
import logging
import gc
import atexit
import itertools
import warnings
import csv
import numpy as np
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
import tensorflow as tf

from os.path import join, isfile, exists, isdir, dirname
from pathlib import Path
from glob import glob
from random import shuffle, choice
from string import ascii_lowercase
from multiprocessing.dummy import Pool as DPool
from functools import partial

import slideflow.model as sfmodel
import slideflow.util as sfutil
import slideflow.io as sfio

from slideflow.io.datasets import Dataset
from slideflow.util import TCGA, ProgressBar, log
from slideflow.activations import ActivationsVisualizer, TileVisualizer, Heatmap
from slideflow.statistics import TFRecordUMAP
from slideflow.mosaic import Mosaic
from comet_ml import Experiment

__version__ = "1.7.0"

NO_LABEL = 'no_label'
SILENT = 'SILENT'
SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))
COMET_API_KEY = "A3VWRcPaHgqc4H5K0FoCtRXbp"

DEFAULT_FLAGS = {
	'use_comet': False,
	'skip_verification': False,
	'eval_batch_size': 64,
	'num_threads': 4
}

def evaluator(outcome_header, model, project_config, results_dict, filters=None, 
				hyperparameters=None, checkpoint=None, eval_k_fold=None, max_tiles_per_slide=0,
				min_tiles_per_slide=0, flags=None):

	if not flags: flags = DEFAULT_FLAGS

	model_root = dirname(model)

	# Load hyperparameters from saved model
	hp_file = hyperparameters if hyperparameters else join(model_root, 'hyperparameters.json')
	hp_data = sfutil.load_json(hp_file)
	hp = sfmodel.HyperParameters()
	hp._load_dict(hp_data['hp'])
	model_name = f"eval-{hp_data['model_name']}-{sfutil.path_to_name(model)}"
	model_type = hp_data['model_type']

	# Filter out slides that are blank in the outcome category
	filter_blank = [outcome_header] if type(outcome_header) != list else outcome_header

	# Load dataset and annotations for evaluation
	eval_dataset = Dataset(config_file=project_config['dataset_config'], sources=project_config['datasets'])
	eval_dataset.load_annotations(project_config['annotations'])
	eval_dataset.apply_filters(filters=filters)
	outcomes, unique_outcomes = eval_dataset.get_outcomes_from_annotations(outcome_header, use_float=(model_type=='linear'))

	# If using a specific k-fold, load validation plan
	if eval_k_fold:
		log.info(f"Using {sfutil.bold('k-fold iteration ' + str(eval_k_fold))}", 1)
		validation_log = join(project_config['root'], "validation_plans.json")
		_, eval_tfrecords = sfio.tfrecords.get_training_and_validation_tfrecords(eval_dataset, validation_log, outcomes, model_type,
																									validation_target=hp_data['validation_target'],
																									validation_strategy=hp_data['validation_strategy'],
																									validation_fraction=hp_data['validation_fraction'],
																									validation_k_fold=hp_data['validation_k_fold'],
																									k_fold_iter=eval_k_fold)
	# Otherwise use all TFRecords
	else:
		eval_tfrecords = eval_dataset.get_tfrecords(ask_to_merge_subdirs=True)

	# Set up model for evaluation
	# Using the project annotation file, assemble list of slides for training, as well as the slide annotations dictionary (output labels)
	model_dir = join(project_config['models_dir'], model_name)

	# Build a model using the slide list as input and the annotations dictionary as output labels
	SFM = sfmodel.SlideflowModel(model_dir, project_config['tile_px'], outcomes, 
																		train_tfrecords=None,
																		validation_tfrecords=eval_tfrecords,
																		manifest=eval_dataset.get_manifest(),
																		use_fp16=project_config['use_fp16'],
																		model_type=model_type)

	# Log model settings and hyperparameters
	hp_file = join(model_dir, 'hyperparameters.json')
	hp_data = {
		"model_name": model_name,
		"model_path": model,
		"stage": "evaluation",
		"tile_px": project_config['tile_px'],
		"tile_um": project_config['tile_um'],
		"model_type": model_type,
		"outcome_headers": outcome_header,
		"outcome_labels": None if model_type != 'categorical' else dict(zip(range(len(unique_outcomes)), unique_outcomes)),
		"dataset_config": project_config['dataset_config'],
		"datasets": project_config['datasets'],
		"annotations": project_config['annotations'],
		"validation_target": hp_data['validation_target'],
		"validation_strategy": hp_data['validation_strategy'],
		"validation_fraction": hp_data['validation_fraction'],
		"validation_k_fold": hp_data['validation_k_fold'],
		"k_fold_i": eval_k_fold,
		"filters": filters,
		"pretrain": None,
		"resume_training": None,
		"checkpoint": checkpoint,
		"comet_experiment": None,
		"hp": hp._get_dict()
	}
	sfutil.write_json(hp_data, hp_file)

	# Perform evaluation
	log.info(f"Evaluating {sfutil.bold(len(eval_tfrecords))} tfrecords", 1)
	
	results = SFM.evaluate(tfrecords=eval_tfrecords, 
						   hp=hp,
						   model=model,
						   model_type=model_type,
						   checkpoint=checkpoint,
						   batch_size=flags['eval_batch_size'],
						   max_tiles_per_slide=max_tiles_per_slide,
						   min_tiles_per_slide=min_tiles_per_slide)

	# Load results into multiprocessing dictionary
	results_dict['results'] = results
	return results_dict

def heatmap_generator(slide, model_name, model_path, save_folder, roi_list, resolution, project_config, flags=None):
	import slideflow.slide as sfslide
	if not flags: flags = DEFAULT_FLAGS

	resolutions = {'low': 1, 'medium': 2, 'high': 4}
	try:
		stride_div = resolutions[resolution]
	except KeyError:
		log.error(f"Invalid resolution '{resolution}': must be either 'low', 'medium', or 'high'.")
		return

	#for slide in slide_list:
	log.empty(f"Working on slide {sfutil.green(sfutil.path_to_name(slide))}", 1)
	heatmap = Heatmap(slide, model_path, project_config['tile_px'], project_config['tile_um'], 
																	use_fp16=project_config['use_fp16'],
																	stride_div=stride_lder,
																	roi_list=roi_list)
	heatmap.generate(batch_size=flags['eval_batch_size'])
	heatmap.save()

def mosaic_generator(model, filters, focus_filters, resolution, num_tiles_x, max_tiles_per_slide, project_config, export_activations=False, flags=None):
	if not flags: flags = DEFAULT_FLAGS

	mosaic_dataset = Dataset(config_file=project_config['dataset_config'], sources=project_config['datasets'])
	mosaic_dataset.load_annotations(project_config['annotations'])
	mosaic_dataset.apply_filters(filters=filters)
	tfrecords_list = mosaic_dataset.get_tfrecords(ask_to_merge_subdirs=True)
	model_path = model if model[-3:] == ".h5" else join(project_config['models_dir'], model, 'trained_model.h5')

	if focus_filters:
		mosaic_dataset.apply_filters(focus_filters)
		focus_list = mosaic_dataset.get_tfrecords(ask_to_merge_subdirs=True)
	else:
		focus_list = None
	log.info(f"Generating mosaic from {len(tfrecords_list)} slides, with focus on {0 if not focus_list else len(focus_list)} slides.", 1)

	AV = ActivationsVisualizer(model=model_path,
							   tfrecords=tfrecords_list, 
							   root_dir=project_config['root'],
							   image_size=project_config['tile_px'],
							   focus_nodes=None,
							   use_fp16=project_config['use_fp16'],
							   batch_size=flags['eval_batch_size'],
							   export_csv=export_activations,
							   max_tiles_per_slide=max_tiles_per_slide)

	AV.generate_mosaic(focus=focus_list,
						num_tiles_x=num_tiles_x,
						resolution=resolution,
						expanded=True)

def trainer(outcome_headers, model_name, model_type, project_config, results_dict, hp, validation_strategy, 
			validation_target, validation_fraction, validation_k_fold, validation_log, validation_dataset=None, 
			validation_annotations=None, validation_filters=None, k_fold_i=None, filters=None, pretrain=None, 
			resume_training=None, checkpoint=None, validate_on_batch=0, max_tiles_per_slide=0, min_tiles_per_slide=0, flags=None):

	if not flags: flags = DEFAULT_FLAGS

	# First, clear prior Tensorflow graph to free memory
	tf.keras.backend.clear_session()

	# Log current model name and k-fold iteration, if applicable
	k_fold_msg = "" if not k_fold_i else f" ({validation_strategy} iteration #{k_fold_i})"
	log.empty(f"Training model {sfutil.bold(model_name)}{k_fold_msg}...")
	log.info(hp, 1)
	full_model_name = model_name if not k_fold_i else model_name+f"-kfold{k_fold_i}"

	# Initialize Comet experiment
	if flags['use_comet']:
		experiment = Experiment(COMET_API_KEY, project_name=project_config['name'])
		experiment.log_parameters(hp._get_dict())
		experiment.log_other('model_name', model_name)
		if k_fold_i:
			experiment.log_other('k_fold_iter', k_fold_i)

	# Load dataset and annotations for training
	training_dataset = Dataset(config_file=project_config['dataset_config'], sources=project_config['datasets'])
	training_dataset.load_annotations(project_config['annotations'])
	training_dataset.apply_filters(filters=filters, filter_blank=outcome_headers)

	# Load outcomes
	outcomes, unique_outcomes = training_dataset.get_outcomes_from_annotations(outcome_headers, use_float=(model_type == 'linear'))
	if model_type == 'categorical': 
		outcome_labels = dict(zip(range(len(unique_outcomes)), unique_outcomes))
	else:
		outcome_labels = dict(zip(range(len(outcome_headers)), outcome_headers))

	# Get TFRecords for training and validation
	manifest = training_dataset.get_manifest()
	training_tfrecords, validation_tfrecords = sfio.tfrecords.get_training_and_validation_tfrecords(training_dataset, validation_log, outcomes, model_type,
																									validation_target=validation_target,
																									validation_strategy=validation_strategy,
																									validation_fraction=validation_fraction,
																									validation_k_fold=validation_k_fold,
																									k_fold_iter=k_fold_i)
	# Use external validation dataset if specified
	if validation_dataset:
		validation_dataset = Dataset(config_file=project_config['dataset_config'], sources=validation_dataset)
		validation_dataset.load_annotations(validation_annotations)
		validation_dataset.apply_filters(filters=validation_filters, filter_blank=outcome_headers)
		validation_tfrecords = validation_dataset.get_tfrecords()
		manifest.update(validation_dataset.get_manifest())
		validation_outcomes, _ = validation_dataset.get_outcomes_from_annotations(outcome_headers, use_float=(model_type == 'linear'))
		outcomes.update(validation_outcomes)

	# Initialize model
	# Using the project annotation file, assemble list of slides for training, as well as the slide annotations dictionary (output labels)
	model_dir = join(project_config['models_dir'], full_model_name)

	# Build a model using the slide list as input and the annotations dictionary as output labels
	SFM = sfmodel.SlideflowModel(model_dir, project_config['tile_px'], outcomes, training_tfrecords, validation_tfrecords,
																			manifest=manifest,
																			use_fp16=project_config['use_fp16'],
																			model_type=model_type)

	# Log model settings and hyperparameters
	hp_file = join(project_config['models_dir'], full_model_name, 'hyperparameters.json')
	hp_data = {
		"model_name": model_name,
		"stage": "training",
		"tile_px": project_config['tile_px'],
		"tile_um": project_config['tile_um'],
		"model_type": model_type,
		"outcome_headers": outcome_headers,
		"outcome_labels": outcome_labels,
		"dataset_config": project_config['dataset_config'],
		"datasets": project_config['datasets'],
		"annotations": project_config['annotations'],
		"validation_target": validation_target,
		"validation_strategy": validation_strategy,
		"validation_fraction": validation_fraction,
		"validation_k_fold": validation_k_fold,
		"k_fold_i": k_fold_i,
		"filters": filters,
		"pretrain": pretrain,
		"resume_training": resume_training,
		"checkpoint": checkpoint,
		"comet_experiment": None if not flags['use_comet'] else experiment.get_key(),
		"hp": hp._get_dict()
	}
	sfutil.write_json(hp_data, hp_file)

	# Execute training
	try:
		results, history = SFM.train(hp, pretrain=pretrain, 
										 resume_training=resume_training, 
										 checkpoint=checkpoint,
										 validate_on_batch=validate_on_batch,
										 max_tiles_per_slide=max_tiles_per_slide,
										 min_tiles_per_slide=min_tiles_per_slide)
		results['history'] = history
		results_dict.update({full_model_name: results})
		logged_epochs = [int(e[5:]) for e in results['epochs'].keys() if e[:5] == 'epoch']
		
		if flags['use_comet']: experiment.log_metrics(results['epochs'][f'epoch{max(logged_epochs)}'])
		del(SFM)
		return history
	except tf.errors.ResourceExhaustedError:
		log.empty("\n")
		log.error(f"Training failed for {sfutil.bold(model_name)}, GPU memory exceeded.", 0)
		del(SFM)
		return None

class SlideflowProject:
	FLAGS = DEFAULT_FLAGS
	GPU_LOCK = None

	def __init__(self, project_folder, interactive=True):
		'''Initializes project by creating project folder, prompting user for project settings, and project
		settings to "settings.json" within the project directory.'''
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
		log.header(f"Slideflow v{__version__}\n================")
		log.header("Loading project...")
		if project_folder and not os.path.exists(project_folder):
			if interactive:
				if sfutil.yes_no_input(f'Directory "{project_folder}" does not exist. Create directory and set as project root? [Y/n] ', default='yes'):
					os.makedirs(project_folder)
				else:
					project_folder = sfutil.dir_input("Where is the project root directory? ", None, create_on_invalid=True, absolute=True)
			else:
				log.info(f"Project directory {project_folder} not found; will create.")
				os.makedirs(project_folder)
		if not project_folder:
			project_folder = sfutil.dir_input("Where is the project root directory? ", None, create_on_invalid=True, absolute=True)

		if exists(join(project_folder, "settings.json")):
			self.load_project(project_folder)
		elif interactive:
			self.create_project(project_folder)

		atexit.register(self.release_gpu)

	def autoselect_gpu(self, number_available, reverse=True):
		'''Automatically claims a free GPU and creates a lock file to prevent 
		other instances of slideflow from using the same GPU.'''

		physical_devices = tf.config.experimental.list_physical_devices('GPU')
		if not len(physical_devices):
			print("No GPUs detected.")
			return
		gpus = range(number_available) if not reverse else reversed(range(number_available))

		for n in gpus:
			if not exists(join(SOURCE_DIR, f"gpu{n}.lock")):
				log.empty(f"Requesting GPU #{n}")
				os.environ["CUDA_VISIBLE_DEVICES"]=str(n)
				open(join(SOURCE_DIR, f"gpu{n}.lock"), 'a').close()
				self.GPU_LOCK = n
				return
		log.error(f"No free GPUs detected; try deleting 'gpu[#].lock' files in the slideflow directory if GPUs are not in use.")

	def release_gpu(self):
		log.empty("Cleaning up...")
		if self.GPU_LOCK != None and exists(join(SOURCE_DIR, f"gpu{self.GPU_LOCK}.lock")):
			log.empty(f"Freeing GPU {self.GPU_LOCK}...")
			os.remove(join(SOURCE_DIR, f"gpu{self.GPU_LOCK}.lock"))

	def select_gpu(self, number):
		log.empty(f"Requesting GPU #{number}")
		self.GPU_LOCK = number
		os.environ["CUDA_VISIBLE_DEVICES"]=str(number)
		
	def _get_hp(self, row, header):
		'''Internal function used to convert a row in the batch_train CSV file into a HyperParameters object.'''
		model_name_i = header.index('model_name')
		args = header[0:model_name_i] + header[model_name_i+1:]
		model_name = row[model_name_i]
		hp = sfmodel.HyperParameters()
		for arg in args:
			value = row[header.index(arg)]
			if arg in hp._get_args():
				if arg != 'finetune_epochs':
					arg_type = type(getattr(hp, arg))
					if arg_type == bool:
						if value.lower() in ['true', 'yes', 'y', 't']:
							bool_val = True
						elif value.lower() in ['false', 'no', 'n', 'f']:
							bool_val = False
						else:
							log.warn(f'Unable to parse arg "{arg}" with value "{value}" in batch training file into true/false; will default to True', 1)
							bool_val = True
						setattr(hp, arg, bool_val)
					else:
						setattr(hp, arg, arg_type(value))
				else:
					epochs = [int(i) for i in value.split(',')]
					setattr(hp, arg, epochs)
			else:
				log.error(f"Unknown argument '{arg}' found in training config file.", 0)
		return hp, model_name

	def _get_hyperparameter_combinations(self, hyperparameters, models, batch_train_file):
		'''Returns list of hyperparameters ojects and associated models names, either from specified hyperparameters or from a batch_train file
		if hyperparameters is None.'''
		if not hyperparameters:
			hp_models_to_train = self._get_valid_models(batch_train_file, models)
		else:
			hp_models_to_train = [models]

		hyperparameter_list = []	
		if not hyperparameters:
			# Assembling list of models and hyperparameters from batch_train.tsv file
			batch_train_rows = []
			with open(batch_train_file) as csv_file:
				reader = csv.reader(csv_file, delimiter='\t')
				header = next(reader)
				for row in reader:
					batch_train_rows += [row]
				
			for row in batch_train_rows:
				# Read hyperparameters
				hp, hp_model_name = self._get_hp(row, header)
				if hp_model_name not in hp_models_to_train: continue

				# Verify HP combinations are valid
				if not self._valid_hp(hp):
					return

				hyperparameter_list += [[hp, hp_model_name]]
		elif (type(hyperparameters) == list) and (type(models) == list):
			if len(models) != len(hyperparameters):
				log.error(f"Unable to iterate through hyperparameters provided; length of hyperparameters ({len(hyperparameters)}) much match length of models ({len(models)})", 1)
				return
			for i in range(len(models)):
				if not self._valid_hp(hyperparameters[i]):
					return
				hyperparameter_list += [[hyperparameters[i], models[i]]]
		else:
			if not self._valid_hp(hyperparameters):
				return
			hyperparameter_list = [[hyperparameters, models]]
		return hyperparameter_list

	def _get_valid_models(self, batch_train_file, models):
		'''Internal function used to scan a batch_train file for valid, trainable models.'''
		models_to_train = []
		with open(batch_train_file) as csv_file:
			reader = csv.reader(csv_file, delimiter="\t")
			header = next(reader)
			try:
				model_name_i = header.index('model_name')
			except:
				log.error("Unable to find column 'model_name' in the batch training config file.", 0)
				sys.exit() 
			for row in reader:
				model_name = row[model_name_i]
				# First check if this row is a valid model
				if (not models) or (type(models)==str and model_name==models) or model_name in models:
					# Now verify there are no duplicate model names
					if model_name in models_to_train:
						log.error(f"Duplicate model names found in {sfutil.green(batch_train_file)}.", 0)
						sys.exit()
					models_to_train += [model_name]
		return models_to_train

	def _valid_hp(self, hp):
		if (hp.model_type() != 'categorical' and ((hp.balanced_training == sfmodel.BALANCE_BY_CATEGORY) or 
											    (hp.balanced_validation == sfmodel.BALANCE_BY_CATEGORY))):
			log.error(f'Invalid hyperparameter combination: balancing type "{sfmodel.BALANCE_BY_CATEGORY}" and model type "{hp.model_type()}".', 1)
			return False
		return True

	def add_dataset(self, name, slides, roi, tiles, tfrecords, label, path=None):
		'''Adds a dataset to the dataset configuration file.

		Args:
			name:		Dataset name.
			slides:		Path to directory containing slides.
			roi:		Path to directory containing CSV ROIs.
			tiles:		Path to directory in which to store extracted tiles.
			tfrecords:	Path to directory in which to store TFRecords of extracted tiles.
			label:		Label to give dataset (typically identifies tile size in pixels/microns).'''

		if not path:
			path = self.PROJECT['dataset_config']
		try:
			datasets_data = sfutil.load_json(path)
		except FileNotFoundError:
			datasets_data = {}
		datasets_data.update({name: {
			'slides': slides,
			'roi': roi,
			'tiles': tiles,
			'tfrecords': tfrecords,
			'label': label
		}})
		sfutil.write_json(datasets_data, path)
		log.info(f"Saved dataset {name} to {path}")

	def associate_slide_names(self):
		'''Experimental function used to automatically associated patient names with slide filenames in the annotations file.'''
		# Load dataset
		dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])

		dataset.update_annotations_with_slidenames(self.PROJECT['annotations'])
		dataset.load_annotations(self.PROJECT['annotations'])

	def create_blank_annotations_file(self, outfile=None):
		'''Creates an example blank annotations file.'''
		if not outfile: 
			outfile = self.PROJECT['annotations']
		with open(outfile, 'w') as csv_outfile:
			csv_writer = csv.writer(csv_outfile, delimiter=',')
			header = [TCGA.patient, 'dataset', 'category']
			csv_writer.writerow(header)

	def create_blank_train_config(self, filename=None):
		'''Creates a CSV file with the batch training structure.'''
		if not filename:
			filename = self.PROJECT['batch_train_config']
		with open(filename, 'w') as csv_outfile:
			writer = csv.writer(csv_outfile, delimiter='\t')
			# Create headers and first row
			header = ['model_name']
			firstrow = ['model1']
			default_hp = sfmodel.HyperParameters()
			for arg in default_hp._get_args():
				header += [arg]
				firstrow += [getattr(default_hp, arg)]
			writer.writerow(header)
			writer.writerow(firstrow)

	def create_hyperparameter_sweep(self, finetune_epochs, toplayer_epochs, model, pooling, loss, learning_rate, batch_size, hidden_layers,
									optimizer, early_stop, early_stop_patience, balanced_training, balanced_validation, augment, 
									hidden_layer_width, trainable_layers, L2_weight, filename=None):
		'''Prepares a hyperparameter sweep using the batch train config file.'''
		log.header("Preparing hyperparameter sweep...")
		# Assemble all possible combinations of provided hyperparameters
		pdict = locals()
		del(pdict['self'])
		del(pdict['filename'])
		del(pdict['finetune_epochs'])
		args = list(pdict.keys())
		for arg in args:
			if type(pdict[arg]) != list:
				pdict[arg] = [pdict[arg]]
		argsv = list(pdict.values())
		sweep = list(itertools.product(*argsv))

		if not filename:
			filename = self.PROJECT['batch_train_config']
		with open(filename, 'w') as csv_outfile:
			writer = csv.writer(csv_outfile, delimiter='\t')
			# Create headers
			header = ['model_name', 'finetune_epochs']
			for arg in args:
				header += [arg]
			writer.writerow(header)
			# Iterate through sweep
			for i, params in enumerate(sweep):
				row = [f'HPSweep{i}', ','.join([str(f) for f in finetune_epochs])]
				full_params = dict(zip(['finetune_epochs'] + args, [finetune_epochs] + list(params)))
				hp = sfmodel.HyperParameters(**full_params)
				for arg in args:
					row += [getattr(hp, arg)]
				writer.writerow(row)
		log.complete(f"Wrote {len(sweep)} combinations for sweep to {sfutil.green(filename)}")

	def create_project(self, project_folder):
		'''Prompts user to provide all relevant project configuration and saves configuration to "settings.json".'''
		# General setup and slide configuration
		project = {
			'root': project_folder,
			'slideflow_version': __version__
		}
		project['name'] = input("What is the project name? ")
		
		# Ask for annotations file location; if one has not been made, offer to create a blank template and then exit
		if not sfutil.yes_no_input("Has an annotations (CSV) file already been created? [y/N] ", default='no'):
			if sfutil.yes_no_input("Create a blank annotations file? [Y/n] ", default='yes'):
				project['annotations'] = sfutil.file_input("Where will the annotation file be located? [./annotations.csv] ", 
									root=project['root'], default='./annotations.csv', filetype="csv", verify=False)
				self.create_blank_annotations_file(project['annotations'])
		else:
			project['annotations'] = sfutil.file_input("Where is the project annotations (CSV) file located? [./annotations.csv] ", 
									root=project['root'], default='./annotations.csv', filetype="csv")

		# Dataset configuration
		project['dataset_config'] = sfutil.file_input("Where is the dataset configuration file located? [./datasets.json] ",
													root=project['root'], default='./datasets.json', filetype='json', verify=False)

		project['datasets'] = []
		while not project['datasets']:
			datasets_data, datasets_names = self.load_datasets(project['dataset_config'])

			print(sfutil.bold("Detected datasets:"))
			if not len(datasets_names):
				print(" [None]")
			else:
				for i, name in enumerate(datasets_names):
					print(f" {i+1}. {name}")
				print(f" {len(datasets_names)+1}. ADD NEW")
				dataset_selection = sfutil.choice_input(f"Which datasets should be used? (choose {len(datasets_names)+1} to add a new dataset) ", valid_choices=[str(l) for l in list(range(1, len(datasets_names)+2))], multi_choice=True)

			if not len(datasets_names) or str(len(datasets_names)+1) in dataset_selection:
				# Create new dataset
				print(f"{sfutil.bold('Creating new dataset')}")
				dataset_name = input("What is the dataset name? ")
				dataset_slides = sfutil.dir_input("Where are the slides stored? [./slides] ",
										root=project['root'], default='./slides', create_on_invalid=True)
				dataset_roi = sfutil.dir_input("Where are the ROI files (CSV) stored? [./slides] ",
										root=project['root'], default='./slides', create_on_invalid=True)
				dataset_tiles = sfutil.dir_input("Where will the tessellated image tiles be stored? (recommend SSD) [./tiles] ",
										root=project['root'], default='./tiles', create_on_invalid=True)
				dataset_tfrecords = sfutil.dir_input("Where should the TFRecord files be stored? (recommend HDD) [./tfrecord] ",
										root=project['root'], default='./tfrecord', create_on_invalid=True)

				self.add_dataset(name=dataset_name,
								 slides=dataset_slides,
								 roi=dataset_roi,
								 tiles=dataset_tiles,
								 tfrecords=dataset_tfrecords,
								 label=NO_LABEL,
								 path=project['dataset_config'])

				print("Updated dataset configuration file.")
			else:
				try:
					project['datasets'] = [datasets_names[int(j)-1] for j in dataset_selection]
				except TypeError:
					print(f'Invalid selection: {dataset_selection}')
					continue

		# Training
		project['models_dir'] = sfutil.dir_input("Where should the saved models be stored? [./models] ",
									root=project['root'], default='./models', create_on_invalid=True)
		project['tile_um'] = sfutil.int_input("What is the tile width in microns? [280] ", default=280)
		project['tile_px'] = sfutil.int_input("What is the tile width in pixels? [224] ", default=224)
		project['use_fp16'] = sfutil.yes_no_input("Should FP16 be used instead of FP32? (recommended) [Y/n] ", default='yes')
		project['batch_train_config'] = sfutil.file_input("Location for the batch training TSV config file? [./batch_train.tsv] ",
													root=project['root'], default='./batch_train.tsv', filetype='tsv', verify=False)
		
		if not exists(project['batch_train_config']):
			print("Batch training file not found, creating blank")
			self.create_blank_train_config(project['batch_train_config'])
		
		# Validation strategy
		project['validation_fraction'] = sfutil.float_input("What fraction of training data should be used for validation testing? [0.2] ", valid_range=[0,1], default=0.2)
		project['validation_target'] = sfutil.choice_input("How should validation data be selected by default, per-tile or per-patient? [per-patient] ", valid_choices=['per-tile', 'per-patient'], default='per-patient')
		if project['validation_target'] == 'per-patient':
			project['validation_strategy'] = sfutil.choice_input("Which validation strategy should be used by default, k-fold, bootstrap, or fixed? [k-fold]", valid_choices=['k-fold', 'bootstrap', 'fixed', 'none'], default='k-fold')
		else:
			project['validation_strategy'] = sfutil.choice_input("Which validation strategy should be used by default, k-fold or fixed? ", valid_choices=['k-fold', 'fixed', 'none'])
		if project['validation_strategy'] == 'k-fold':
			project['validation_k_fold'] = sfutil.int_input("What is K? [3] ", default=3)
		elif project['validation_strategy'] == 'bootstrap':
			project['validation_k_fold'] = sfutil.int_input("How many iterations should be performed when bootstrapping? [3] ", default=3)
		else:
			project['validation_k_fold'] = 0

		sfutil.write_json(project, join(project_folder, 'settings.json'))
		self.PROJECT = project

		# Write a sample actions.py file
		with open(join(SOURCE_DIR, 'sample_actions.py'), 'r') as sample_file:
			sample_actions = sample_file.read()
			with open(os.path.join(project_folder, 'actions.py'), 'w') as actions_file:
				actions_file.write(sample_actions)

		print("\nProject configuration saved.\n")
		self.load_project(project_folder)

	def evaluate(self, model, outcome_header, hyperparameters=None, filters=None, checkpoint=None,
					eval_k_fold=None, max_tiles_per_slide=0, min_tiles_per_slide=0):
		'''Evaluates a saved model on a given set of tfrecords.
		
		Args:
			model:					Path to .h5 model to evaluate.
			outcome_header:			Annotation column header that specifies the outcome label.
			hyperparameters:		Path to model's hyperparameters.json file. If None, searches for this file in the same directory as the model.
			filters:				Filters to use when selecting tfrecords on which to perform evaluation.
			checkpoint:				Path to cp.ckpt file to load, if evaluating a saved checkpoint.
			eval_k_fold:			K-fold iteration number to evaluate. If None, will evaluate all tfrecords irrespective of K-fold.
			max_tiles_per_slide:	Will only use up to this many tiles from each slide for evaluation. If zero, will include all tiles.
			min_tiles_per_slide:	Minimum number of tiles a slide must have to be included in evaluation. Default is 0, but
										for best slide-level AUC, a minimum of at least 10 tiles per slide is recommended.'''
										
		log.header(f"Evaluating model {sfutil.green(model)}...")

		manager = multiprocessing.Manager()
		results_dict = manager.dict()
		ctx = multiprocessing.get_context('spawn')
		
		process = ctx.Process(target=evaluator, args=(outcome_header, model, self.PROJECT, results_dict, filters, hyperparameters, 
														checkpoint, eval_k_fold, max_tiles_per_slide, min_tiles_per_slide, self.FLAGS))
		process.start()
		log.empty(f"Spawning evaluation process (PID: {process.pid})")
		process.join()

		return results_dict

	def extract_dual_tiles(self, tile_um=None, tile_px=None, stride_div=1, filters=None):
		import slideflow.slide as sfslide
		from PIL import Image

		# Filter out warnings and allow loading large images
		warnings.simplefilter('ignore', Image.DecompressionBombWarning)
		Image.MAX_IMAGE_PIXELS = 100000000000

		tile_um = self.PROJECT['tile_um'] if not tile_um else tile_um
		tile_px = self.PROJECT['tile_px'] if not tile_px else tile_px

		log.header("Extracting dual-image tiles...")
		extracting_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		extracting_dataset.load_annotations(self.PROJECT['annotations'])
		extracting_dataset.apply_filters(filters=filters)	

		def extract_tiles_from_slide(slide_path, roi_list, dataset_config, pb):
			root_path = join(dataset_config["tfrecords"], dataset_config["label"])
			if not exists(root_path): 
					os.makedirs(root_path)

			whole_slide = sfslide.SlideReader(slide_path, tile_px, tile_um, stride_div, roi_list=roi_list, pb=pb)
			small_tile_generator, _, _, _ = whole_slide.build_generator(dual_extract=True)
			tfrecord_name = sfutil.path_to_name(slide_path)
			tfrecord_path = join(root_path, f"{tfrecord_name}.tfrecords")
			records = []

			for image_dict in small_tile_generator():
				label = bytes(tfrecord_name, 'utf-8')
				image_string_dict = {}
				for image_label in image_dict:
					np_image = image_dict[image_label]
					image = Image.fromarray(np_image).convert('RGB')
					with io.BytesIO() as output:
						image.save(output, format="JPEG")
						image_string = output.getvalue()
						image_string_dict.update({
							image_label: image_string
						})
				records += [[label, image_string_dict]]

			shuffle(records)
			
			with tf.io.TFRecordWriter(tfrecord_path) as writer:
				for label, image_string_dict in records:
					tf_example = sfio.tfrecords.multi_image_example(label, image_string_dict)
					writer.write(tf_example.SerializeToString())

		for dataset_name in self.PROJECT['datasets']:
			log.empty(f"Working on dataset {sfutil.bold(dataset_name)}", 1)
			slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)
			roi_list = extracting_dataset.get_rois()
			dataset_config = extracting_dataset.datasets[dataset_name]
			log.info(f"Extracting tiles from {len(slide_list)} slides ({tile_um} um, {tile_px} px)", 1)

			log.info("Exporting tiles only", 1)
			pb = ProgressBar(bar_length=5, counter_text='tiles')

			if self.FLAGS['num_threads'] > 1:
				pool = DPool(self.FLAGS['num_threads'])
				pool.map(partial(extract_tiles_from_slide, roi_list=roi_list, dataset_config=dataset_config, pb=pb), slide_list)
				pool.close()
			else:
				for slide_path in slide_list:
					extract_tiles_from_slide(slide_path, roi_list, dataset_config, pb)
		
		self.update_manifest()

	def extract_tiles(self, tile_um=None, tile_px=None, filters=None, generate_tfrecords=True, stride_div=1, tma=False, augment=False, delete_tiles=True, enable_downsample=False):
		'''Extract tiles from a group of slides; save a percentage of tiles for validation testing if the 
		validation target is 'per-patient'; and generate TFRecord files from the raw images.
		
		Args:
			tile_um:			Tile size in microns. If None, will use project default.
			tile_px:			Tile size in pixels. If None, will use project default.
			filters:			Dataset filters to use when selecting slides for tile extraction.
			generate_tfrecords:	Bool. If True, will export extracted tiles to TFRecords (required for training).
			stride_div:			Stride divisor to use when extracting tiles. A stride of 1 will extract non-overlapping tiles. 
									A stride_div of 2 will extract overlapping tiles, with a stride equal to 50% of the tile width.
			tma:				Bool. If True, reads slides as Tumor Micro-Arrays (TMAs), detecting and extracting tumor cores.
			augment:			Bool. If True, will augment tiles 8-fold by randomly flipping/rotating extracted tiles.
			delete_tiles:		Bool. If True, will delete loose tile images after storing into TFRecords.
			enable_downsample:	Bool. If True, enables the use of downsampling while reading slide images. This may result in corrupted image tiles
									if downsampled slide layers are corrupted or not fully generated. Manual confirmation of tile integrity is recommended.'''

		import slideflow.slide as sfslide
		from PIL import Image

		log.header("Extracting image tiles...")
		tile_um = self.PROJECT['tile_um'] if not tile_um else tile_um
		tile_px = self.PROJECT['tile_px'] if not tile_px else tile_px

		# Load dataset for evaluation
		extracting_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		extracting_dataset.load_annotations(self.PROJECT['annotations'])
		extracting_dataset.apply_filters(filters=filters)

		for dataset_name in self.PROJECT['datasets']:
			log.empty(f"Working on dataset {sfutil.bold(dataset_name)}", 1)
			slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)
			log.info(f"Extracting tiles from {len(slide_list)} slides ({tile_um} um, {tile_px} px)", 1)
			
			save_folder = join(extracting_dataset.datasets[dataset_name]['tiles'], extracting_dataset.datasets[dataset_name]['label'])
			roi_dir = extracting_dataset.datasets[dataset_name]['roi']

			if not os.path.exists(save_folder):
				os.makedirs(save_folder)

			# Filter out warnings and allow loading large images
			warnings.simplefilter('ignore', Image.DecompressionBombWarning)
			Image.MAX_IMAGE_PIXELS = 100000000000

			log.info("Exporting tiles only", 1)
			pb = ProgressBar(bar_length=5, counter_text='tiles')

			# Function to extract tiles from a slide
			def extract_tiles_from_slide(slide_path, pb):
				log.empty(f"Exporting tiles for slide {sfutil.path_to_name(slide_path)}", 1, pb.print)

				if not tma:
					whole_slide = sfslide.SlideReader(slide_path, tile_px, tile_um, stride_div, enable_downsample=enable_downsample, 
																								export_folder=save_folder,
																								roi_dir=roi_dir,
																								roi_list=None, pb=pb)
				else:
					whole_slide = sfslide.TMAReader(slide_path, tile_px, tile_um, stride_div, enable_downsample=enable_downsample,
																							  export_folder=save_folder, 
																							  pb=pb)

				if not whole_slide.loaded_correctly():
					return

				whole_slide.export_tiles(augment=augment)

			# Use multithreading if specified, extracting tiles from all slides in the filtered list
			if self.FLAGS['num_threads'] > 1:
				pool = DPool(self.FLAGS['num_threads'])
				pool.map(partial(extract_tiles_from_slide, pb=pb), slide_list)
				pool.close()
			else:
				for slide_path in slide_list:
					extract_tiles_from_slide(slide_path, pb)

			# Now, split extracted tiles into validation/training subsets if per-tile validation is being used
			if self.PROJECT['validation_target'] == 'per-tile':
				if self.PROJECT['validation_target'] == 'per-tile':
					if self.PROJECT['validation_strategy'] == 'boostrap':
						log.warn("Validation bootstrapping is not supported when the validation target is per-tile; will generate random fixed validation target", 1)
					if self.PROJECT['validation_strategy'] in ('bootstrap', 'fixed'):
						# Split the extracted tiles into two groups
						sf.datasets.split_tiles(save_folder, fraction=[-1, self.PROJECT['validation_fraction']], names=['training', 'validation'])
					if self.PROJECT['validation_strategy'] == 'k-fold':
						sf.datasets.split_tiles(save_folder, fraction=[-1] * self.PROJECT['validation_k_fold'], names=[f'kfold-{i}' for i in range(self.PROJECT['validation_k_fold'])])

		# Generate TFRecords from the extracted tiles
		if generate_tfrecords:
			self.generate_tfrecords_from_tiles(delete_tiles)

	def generate_activations_analytics(self, model, outcome_header, filters=None, focus_nodes=[], node_exclusion=False):
		'''Calculates final layer activations and displays information regarding the most significant final layer nodes.
		
		Note: GPU memory will remain in use, as the Keras model associated with the visualizer is active.'''
		log.header("Generating final layer activation analytics...")

		# Load dataset for evaluation
		activations_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		activations_dataset.load_annotations(self.PROJECT['annotations'])
		activations_dataset.apply_filters(filters=filters)
		tfrecords_list = activations_dataset.get_tfrecords(ask_to_merge_subdirs=True)
		model_path = model if model[-3:] == ".h5" else join(self.PROJECT['models_dir'], model, 'trained_model.h5')
		log.info(f"Visualizing activations from {len(tfrecords_list)} slides", 1)

		AV = ActivationsVisualizer(model=model_path,
								   tfrecords=tfrecords_list,
								   root_dir=self.PROJECT['root'],
								   image_size=self.PROJECT['tile_px'],
								   annotations=self.PROJECT['annotations'],
								   outcome_header=outcome_header,
								   focus_nodes=focus_nodes,
								   use_fp16=self.PROJECT['use_fp16'])

		return AV

	def generate_heatmaps(self, model, filters=None, resolution='low'):
		'''Creates predictive heatmap overlays on a set of slides. 

		Args:
			model:			Path to .h5 model with which predictions will be generated.
			filters:		Dataset filters to use when selecting slides for which to generate heatmaps.
			resolution:		Heatmap resolution (determines stride of tile predictions). 
								"low" uses a stride equal to tile width.
								"medium" uses a stride equal 1/2 tile width.
								"high" uses a stride equal to 1/4 tile width.
		'''
		log.header("Generating heatmaps...")

		# Prepare dataset
		heatmaps_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		heatmaps_dataset.load_annotations(self.PROJECT['annotations'])
		heatmaps_dataset.apply_filters(filters=filters)
		slide_list = heatmaps_dataset.get_slide_paths()
		roi_list = heatmaps_dataset.get_rois()
		model_path = model if model[-3:] == ".h5" else join(self.PROJECT['models_dir'], model, 'trained_model.h5')

		# Attempt to auto-detect supplied model name
		detected_model_name = sfutil.path_to_name(model_path)
		hp_file = join(*model_path.split('/')[:-1], 'hyperparameters.json')
		if exists(hp_file):
			loaded_hp = sfutil.load_json(hp_file)
			if 'model_name' in loaded_hp:
				detected_model_name = loaded_hp['model_name']
		
		# Make output directory
		heatmaps_folder = os.path.join(self.PROJECT['root'], 'heatmaps', detected_model_name)
		if not exists(heatmaps_folder): os.makedirs(heatmaps_folder)

		# Heatmap processes
		ctx = multiprocessing.get_context('spawn')
		for slide in slide_list:
			process = ctx.Process(target=heatmap_generator, args=(slide, model, model_path, heatmaps_folder, roi_list, resolution, self.PROJECT, self.FLAGS))
			process.start()
			log.empty(f"Spawning heatmaps process (PID: {process.pid})")
			process.join()

	def generate_mosaic(self, model, filters=None, focus_filters=None, resolution="low", num_tiles_x=50, max_tiles_per_slide=100, export_activations=False):
		'''Generates a mosaic map with dimensionality reduction on penultimate layer activations. Tile data is extracted from the provided
		set of TFRecords and predictions are calculated using the specified model.
		
		Args:
			model:					Path to .h5 file to use when generating layer activations.
			filters:				Dataset filters to use when selecting slides to include the mosaic.
			focus_filters:			Dataset filters to use when selecting slides to highlight on the mosaic.
			resolution:				Resolution of the mosaic map. Impacts size of the final figure. Either low, medium, or high.
			num_tiles_x:			Specifies the size of the mosaic map grid.
			max_tiles_per_slide:	Limits the number of tiles taken from each slide. Too high of a number may introduce memory issues.
			export_activations:		Bool. If true, will save calculated layer activations to a CSV.'''

		log.header("Generating mosaic map...")
		ctx = multiprocessing.get_context('spawn')
		process = ctx.Process(target=mosaic_generator, args=(model, filters, focus_filters, resolution, num_tiles_x, max_tiles_per_slide, self.PROJECT, export_activations, self.FLAGS))
		process.start()
		log.empty(f"Spawning mosaic process (PID: {process.pid})")
		process.join()

	def generate_mosaic_from_annotations(self, header_x, header_y, header_category=None, filters=None, focus_filters=None, resolution='low', num_tiles_x=50,
											expanded=False, use_optimal_tile=False, model=None, max_tiles_per_slide=0, map_predictions=False):

		dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		dataset.load_annotations(self.PROJECT['annotations'])
		dataset.apply_filters(filters=filters, filter_blank=[header_x, header_y])
		tfrecords = dataset.get_tfrecords()
		slides = dataset.get_slides()
		outcomes, _ = dataset.get_outcomes_from_annotations([header_x, header_y], use_float=True)
		outcomes_category, unique_outcomes = dataset.get_outcomes_from_annotations(header_category)
		slide_to_category = {k:unique_outcomes[v['outcome']] for k, v in outcomes_category.items()}

		umap_x = np.array([outcomes[slide]['outcome'][0] for slide in slides])
		umap_y = np.array([outcomes[slide]['outcome'][1] for slide in slides])

		if use_optimal_tile and not model:
			log.error("Unable to calculate optimal tile if no model is specified.")
			return
		elif use_optimal_tile:
			# Calculate most representative tile in each slide/TFRecord for display
			AV = ActivationsVisualizer(model=model,
									   tfrecords=tfrecords, 
									   root_dir=self.PROJECT['root'],
									   image_size=self.PROJECT['tile_px'],
									   use_fp16=self.PROJECT['use_fp16'],
									   batch_size=self.FLAGS['eval_batch_size'],
									   max_tiles_per_slide=max_tiles_per_slide)

			if map_predictions:
				umap_x, umap_y, umap_meta = AV.get_mapped_predictions()
			else:
				optimal_slide_indices = AV.calculate_centroid_indices()

				# Restrict mosaic to only slides that had enough tiles to calculate an optimal index from centroid
				successful_slides = list(optimal_slide_indices.keys())
				num_warned = 0
				warn_threshold = 3
				for slide in slides:
					print_func = print if num_warned < warn_threshold else None
					if slide not in successful_slides:
						log.warn(f"Unable to calculate optimal tile for slide {sfutil.green(slide)}; will not include in Mosaic", 1, print_func)
						num_warned += 1
				if num_warned >= warn_threshold:
					log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)

				umap_x = np.array([outcomes[slide]['outcome'][0] for slide in successful_slides])
				umap_y = np.array([outcomes[slide]['outcome'][1] for slide in successful_slides])
				umap_meta = [{'slide': slide, 'index': optimal_slide_indices[slide]} for slide in successful_slides]
		else:
			# Take the first tile from each slide/TFRecord
			umap_meta = [{'slide': slide, 'index': 0} for slide in slides]

		umap = TFRecordUMAP(tfrecords=dataset.get_tfrecords(), slides=dataset.get_slides())
		umap.load_precalculated(umap_x, umap_y, umap_meta)

		mosaic_map = Mosaic(umap, leniency=1.5,
								  expanded=expanded,
								  tile_zoom=15,
								  num_tiles_x=num_tiles_x,
								  resolution=resolution)

		mosaic_map.save(join(self.PROJECT['root'], 'stats'))
		umap.save_2d_plot(join(self.PROJECT['root'], 'stats', '2d_mosaic_umap.png'), slide_to_category)

	def generate_tfrecords_from_tiles(self, delete_tiles=True):
		'''Create tfrecord files from a collection of raw images'''
		log.header('Writing TFRecord files...')

		# Load dataset for evaluation
		working_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		
		for d in working_dataset.datasets:
			log.empty(f"Working on dataset {d}", 1)
			config = working_dataset.datasets[d]
			tfrecord_dir = join(config["tfrecords"], config["label"])
			tiles_dir = join(config["tiles"], config["label"])

			# Check to see if subdirectories in the target folders are slide directories (contain images)
			#  or are further subdirectories (e.g. validation and training)
			log.info('Scanning tile directory structure...', 2)
			if sfutil.contains_nested_subdirs(tiles_dir):
				subdirs = [_dir for _dir in os.listdir(tiles_dir) if isdir(join(tiles_dir, _dir))]
				for subdir in subdirs:
					tfrecord_subdir = join(tfrecord_dir, subdir)
					sfio.tfrecords.write_tfrecords_multi(join(tiles_dir, subdir), tfrecord_subdir)
			else:
				sfio.tfrecords.write_tfrecords_multi(tiles_dir, tfrecord_dir)

			self.update_manifest()

			if delete_tiles:
				shutil.rmtree(tiles_dir)

	def load_datasets(self, path):
		'''Loads datasets from a given datasets.json file.'''
		try:
			datasets_data = sfutil.load_json(path)
			datasets_names = list(datasets_data.keys())
			datasets_names.sort()
		except FileNotFoundError:
			datasets_data = {}
			datasets_names = []
		return datasets_data, datasets_names

	def load_project(self, directory):
		'''Loads a saved and pre-configured project.'''
		if exists(join(directory, "settings.json")):
			self.PROJECT = sfutil.load_json(join(directory, "settings.json"))
			log.empty("Project configuration loaded.")
		else:
			raise OSError(f'Unable to locate settings.json at location "{directory}".')

		# Enable logging
		log.logfile = join(self.PROJECT['root'], "log.log")

		# Load dataset for evaluation
		try:
			dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
			dataset.load_annotations(self.PROJECT['annotations'])

			if not self.FLAGS['skip_verification']:
				log.header("Verifying Annotations...")
				dataset.verify_annotations_slides()
				log.header("Verifying TFRecord manifest...")
				self.update_manifest()
		except FileNotFoundError:
			log.warn("No datasets configured.")

	def resize_tfrecords(self, size, filters=None):
		'''Resizes TFRecords to a given pixel size.'''
		log.header(f"Resizing TFRecord tiles to ({size}, {size})")
		resize_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		resize_dataset.load_annotations(self.PROJECT['annotations'])
		resize_dataset.apply_filters(filters=filters)
		tfrecords_list = resize_dataset.get_tfrecords()
		log.info(f"Resizing {len(tfrecords_list)} tfrecords", 1)

		for tfr in tfrecords_list:
			sfio.tfrecords.transform_tfrecord(tfr, tfr+".transformed", resize=size)
	
	def extract_tiles_from_tfrecords(self, destination=None, filters=None):
		'''Extracts all tiles from a set of TFRecords'''
		log.header(f"Extracting tiles from TFRecords")
		to_extract_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		to_extract_dataset.load_annotations(self.PROJECT['annotations'])
		to_extract_dataset.apply_filters(filters)
		
		for dataset_name in self.PROJECT['datasets']:
			to_extract_tfrecords = to_extract_dataset.get_tfrecords(dataset=dataset_name)
			if destination:
				tiles_dir = destination
			else:
				tiles_dir = join(to_extract_dataset.datasets[dataset_name]['tiles'], to_extract_dataset.datasets[dataset_name]['label'])
				if not exists(tiles_dir):
					os.makedirs(tiles_dir)
			for tfr in to_extract_tfrecords:
				sfio.tfrecords.extract_tiles(tfr, tiles_dir)		

	def save_project(self):
		'''Saves current project configuration as "settings.json".'''
		sfutil.write_json(self.PROJECT, join(self.PROJECT['root'], 'settings.json'))

	def train(self, models=None, outcome_header='category', multi_outcome=False, filters=None, resume_training=None, checkpoint=None, 
				pretrain='imagenet', batch_file=None, hyperparameters=None, model_type='categorical',
				validation_target=None, validation_strategy=None, validation_fraction=None, validation_k_fold=None, k_fold_iter=None,
				validation_dataset=None, validation_annotations=None, validation_filters=None, validate_on_batch=256,
				max_tiles_per_slide=0, min_tiles_per_slide=0):
		'''Train model(s) given configurations found in batch_train.tsv.

		Args:
			models:					Either a string representing a model name, or an array of strings containing multiple model names. 
										Required if training to a single hyperparameter combination with the "hyperparameters" argument.
										If performing a hyperparameter sweep, will only train models with these names in the batch_train.tsv config file.
										May supply None if performing a hyperparameter sweep, in which case all models in the batch_train.tsv config file will be trained.
			outcome_header:			String or list. Specifies which header(s) in the annotation file to use for the output category. 
										Defaults to 'category'.	If a list is provided, will loop through all outcomes and perform HP sweep on each.
			multi_outcome:			If True, will train to multiple outcomes simultaneously instead of looping through the
										list of outcomes in "outcome_header". Defaults to False.
			filters:				Dictionary of column names mapping to column values by which to filter slides using the annotation file.
			resume_training:		Path to .h5 model to continue training
			checkpoint:				Path to cp.ckpt from which to load weights
			pretrain:				Pretrained weights to load. Default is imagenet. May supply a compatible .h5 file from which to load weights.
			batch_file:				Manually specify batch file to use for a hyperparameter sweep. If not specified, will use project default.
			hyperparameters:		Manually specify hyperparameter combination to use for training. If specified, will ignore batch training file.
			model_type:				Type of output variable, either categorical (default) or linear.
			validation_target: 		Whether to select validation data on a 'per-patient' or 'per-tile' basis. If not specified, will use project default.
			validation_strategy:	Validation dataset selection strategy (bootstrap, k-fold, fixed, none). If not specified, will use project default.
			validation_fraction:	Fraction of data to use for validation testing. If not specified, will use project default.
			validation_k_fold: 		K, if using k-fold validation. If not specified, will use project default.
			k_fold_iter:			Which iteration to train if using k-fold validation. Defaults to training all iterations.
			validation_dataset:		If specified, will use a separate dataset on which to perform validation.
			validation_annotations:	If using a separate dataset for validation, the annotations CSV must be supplied.
			validation_filters:		If using a separate dataset for validation, these filters are used to select a subset of slides for validation.
			validate_on_batch:		Validation will be performed every X batches.
			max_tiles_per_slide:	Will only use up to this many tiles from each slide for training. If zero, will include all tiles.
			min_tiles_per_slide:	Minimum number of tiles a slide must have to be included in training. 

		Returns:
			A dictionary containing model names mapped to train_acc, val_loss, and val_acc
		'''
		# Reconcile provided arguments with project defaults
		batch_train_file = self.PROJECT['batch_train_config'] if not batch_file else join(self.PROJECT['root'], batch_file)
		validation_strategy = self.PROJECT['validation_strategy'] if not validation_strategy else validation_strategy
		validation_target = self.PROJECT['validation_target'] if not validation_target else validation_target
		validation_fraction = self.PROJECT['validation_fraction'] if not validation_fraction else validation_fraction
		validation_k_fold = self.PROJECT['validation_k_fold'] if not validation_k_fold else validation_k_fold
		validation_log = join(self.PROJECT['root'], "validation_plans.json")

		# Quickly scan for errors (duplicate model names in batch training file) and prepare models to train
		if multi_outcome and model_type != "linear":
			log.error("Multiple outcome variables only supported for linear outcome variables.")
			sys.exit()

		if hyperparameters and not models:
			log.error("If specifying hyperparameters, 'models' must be supplied. ", 1)
			return

		# Prepare hyperparameters
		log.header("Performing hyperparameter sweep...")
		
		hyperparameter_list = self._get_hyperparameter_combinations(hyperparameters, models, batch_train_file)

		outcome_header = [outcome_header] if type(outcome_header) != list else outcome_header
		if multi_outcome:
			log.info(f"Training ({len(hyperparameter_list)} models) using {len(outcome_header)} variables as simultaneous input:", 1)
		else:
			log.header(f"Training ({len(hyperparameter_list)} models) for each of {len(outcome_header)} outcome variables:", 1)
		for outcome in outcome_header:
			log.empty(outcome, 2)
		print()
		outcome_header = [outcome_header] if multi_outcome else outcome_header

		# Prepare k-fold validation configuration
		results_log_path = os.path.join(self.PROJECT['root'], "results_log.csv")
		k_fold_iter = [k_fold_iter] if (k_fold_iter != None and type(k_fold_iter) != list) else k_fold_iter
		k_fold = validation_k_fold if validation_strategy in ('k-fold', 'bootstrap') else 0
		valid_k = [] if not k_fold else [kf for kf in range(1, k_fold+1) if ((k_fold_iter and kf in k_fold_iter) or (not k_fold_iter))]

		# Next, prepare the multiprocessing manager (needed to free VRAM after training and keep track of results)
		manager = multiprocessing.Manager()
		results_dict = manager.dict()
		ctx = multiprocessing.get_context('spawn')

		# If using multiple outcomes, initiate hyperparameter sweep for each outcome category specified
		# If not training to multiple outcome, perform full hyperparameter sweep of the combined outcomes
		for selected_outcome_headers in outcome_header:
			# For each hyperparameter combination, perform training
			for hp, hp_model_name in hyperparameter_list:
				# Generate model name
				if type(selected_outcome_headers) == list:
					outcome_string = "-".join(selected_outcome_headers)
				else:
					outcome_string = selected_outcome_headers
					selected_outcome_headers = [selected_outcome_headers]

				model_name = f"{outcome_string}-{hp_model_name}"
				model_iterations = [model_name] if not k_fold else [f"{model_name}-kfold{k}" for k in valid_k]

				def start_training_process(k):
					# Using a separate process ensures memory is freed once training has completed
					process = ctx.Process(target=trainer, args=(selected_outcome_headers, model_name, model_type, 
																self.PROJECT, results_dict, hp, validation_strategy, 
																validation_target, validation_fraction, validation_k_fold, 
																validation_log, validation_dataset, validation_annotations,
																validation_filters, k, filters, pretrain, resume_training, 
																checkpoint, validate_on_batch, max_tiles_per_slide, min_tiles_per_slide, self.FLAGS))
					process.start()
					log.empty(f"Spawning training process (PID: {process.pid})")
					process.join()

				# Perform training
				if k_fold:
					for k in valid_k:
						start_training_process(k)
						
				else:
					start_training_process(None)

				# Record results
				for mi in model_iterations:
					if mi not in results_dict:
						log.error(f"Training failed for model {model_name} for an unknown reason")
					else:
						sfutil.update_results_log(results_log_path, mi, results_dict[mi]['epochs'])
				log.complete(f"Training complete for model {model_name}, results saved to {sfutil.green(results_log_path)}")

			# Print summary of all models
			log.complete("Training complete; validation accuracies:", 0)
			for model in results_dict:
				try:
					last_epoch = max([int(e.split('epoch')[-1]) for e in results_dict[model]['epochs'].keys() if 'epoch' in e ])
					final_metrics = results_dict[model]['epochs'][f'epoch{last_epoch}']
					log.empty(f" - {sfutil.green(model)}: Train_Acc={str(final_metrics['train_acc'])}, " +
						f"Val_loss={final_metrics['val_loss']}, Val_Acc={final_metrics['val_acc']}" )
				except ValueError:
					pass

		return results_dict

	def update_manifest(self, force_update=False):
		'''Updates manifest file in the TFRecord directory, used to track number of records and verify annotations.
		
		Args:
			force_update:	If True, will re-validate contents of all TFRecords. If False, will only validate
								contents of TFRecords not yet in the manifest
		'''
		dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		dataset.load_annotations(self.PROJECT['annotations'])
		tfrecords_folders = dataset.get_tfrecords_folders()
		for tfr_folder in tfrecords_folders:
			dataset.update_tfrecord_manifest(directory=tfr_folder, 
											 force_update=force_update)

	def update_tfrecords(self):
		log.header('Updating TFRecords...')
		working_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		working_dataset.load_annotations(self.PROJECT['annotations'])

		for d in working_dataset.datasets:
			config = working_dataset.datasets[d]
			tfrecord_folder = join(config["tfrecords"], config["label"])
			num_updated = 0
			log.info(f"Updating TFRecords in {sfutil.green(tfrecord_folder)}...")
			num_updated += sfio.tfrecords.update_tfrecord_dir(tfrecord_folder, slide='case', image_raw='image_raw')
		log.complete(f"Updated {sfutil.bold(num_updated)} TFRecords files")
	
	def visualize_tiles(self, model, directory, node, num_to_visualize=20, window=None):
		tiles = [o for o in os.listdir(directory) if not isdir(join(directory, o))]
		tiles.sort(key=lambda x: int(x.split('-')[0]))
		tiles.reverse()

		TV = TileVisualizer(model=model, 
							node=node,
							shape=[self.PROJECT['tile_px'], self.PROJECT['tile_px'], 3],
							tile_width=window)

		for tile in tiles[:20]:
			tile_loc = join(directory, tile)
			TV.visualize_tile(tile_loc, save_dir=directory)