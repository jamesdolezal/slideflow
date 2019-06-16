# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, March 2019
# ==========================================================================

'''This module includes tools to convolutionally section whole slide images into tiles
using python Generators. These tessellated tiles can be exported as JPGs, with or without
data augmentation, or used as input for a trained Tensorflow model. Model predictions 
can then be visualized as a heatmap overlay.

This module is compatible with SVS and JPG images.

Requires: Openslide (https://openslide.org/download/).'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import warnings
from os.path import join, isfile, exists

import progress_bar

import tensorflow as tf
import numpy as np
import imageio
#import inception_v4
#from tensorflow.contrib.framework import arg_scope
#from inception_utils import inception_arg_scope
from PIL import Image
import argparse
import pickle
import csv
import openslide as ops
import shapely.geometry as sg
import cv2
import json
import time
from math import sqrt

from multiprocessing import Pool
from queue import Queue
from threading import Thread

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcol
from matplotlib import pyplot as mp

from fastim import FastImshow
from util import sfutil

Image.MAX_IMAGE_PIXELS = 100000000000
NUM_THREADS = 4
DEFAULT_JPG_MPP = 0.2494
JSON_ANNOTATION_SCALE = 10

STRICT_AUGMENTATION = "strict"
BALANCED_AUGMENTATION = "balanced"
NO_AUGMENTATION = None

# TODO: offset heatmap by window / 2
# TODO: test json annotations
# TODO: automatic augmentation balancing

class ROIObject:
	'''Object container for ROI annotations.'''
	def __init__(self, name):
		self.name = name
		self.coordinates = []
	def add_coord(self, coord):
		self.coordinates.append(coord)
	def scaled_area(self, scale):
		return np.multiply(self.coordinates, 1/scale)
	def print_coord(self):
		for c in self.coordinates: print(c)
	def add_shape(self, shape):
		for point in shape:
			self.add_coord(point)

class JPGSlide:
	'''Object that provides cross-compatibility with certain OpenSlide methods when using JPG slides.'''
	def __init__(self, path, mpp):
		self.loaded_image = imageio.imread(path)
		self.dimensions = (self.loaded_image.shape[1], self.loaded_image.shape[0])
		self.properties = {ops.PROPERTY_NAME_MPP_X: mpp}
		self.level_dimensions = [self.dimensions]
		self.level_count = 1
		self.level_downsamples = [1.0]

	def get_thumbnail(self, dimensions):
		return cv2.resize(self.loaded_image, dsize=dimensions, interpolation=cv2.INTER_CUBIC)

	def read_region(self, topleft, level, window):
		# Arg "level" required for code compatibility with SVS reader but is not used
		# Window = [y, x] pixels (note: this is reverse compared to SVS files in [x,y] format)
		return self.loaded_image[topleft[1]:topleft[1] + window[1], 
								 topleft[0]:topleft[0] + window[0],]

	def get_best_level_for_downsample(self, downsample_desired):
		return 0

class SlideReader:
	'''Helper object that loads a slide and its ROI annotations and sets up a tile generator.'''
	def __init__(self, path, filetype, export_folder=None, roi_dir=None, pb=None):
		self.print = print if not pb else pb.print
		self.annotations = []
		self.export_folder = export_folder
		self.pb = pb # Progress bar
		self.p_id = None
		self.name = path[:-4].split('/')[-1]
		self.shortname = sfutil._shortname(self.name)
		# Initiate SVS or JPG slide reader
		if filetype == "svs":
			try:
				self.slide = ops.OpenSlide(path)
			except ops.lowlevel.OpenSlideUnsupportedFormatError:
				self.print(f" + {sfutil.warn('[WARN]')}" + f" Unable to read SVS file from {path} , skipping")
				self.shape = None
				return None
		elif filetype == "jpg":
			self.slide = JPGSlide(path, mpp=DEFAULT_JPG_MPP)
		else:
			self.print(f'Unsupported file type "{filetype}" for case {self.shortname}.')
			return None
		
		thumbs_path = join('/'.join(path.split('/')[:-1]), "thumbs")
		if not os.path.exists(thumbs_path): os.makedirs(thumbs_path)

		# Load ROI from roi_dir if available
		if roi_dir and exists(join(roi_dir, self.name + ".csv")):
			self.load_csv_roi(join(roi_dir, self.name + ".csv"))
		# Else try loading ROI from same folder as SVS
		elif exists(path[:-4] + ".csv"):
			self.load_csv_roi(path[:-4] + ".csv")
		else:
			self.print(f"   {sfutil.warn('!')} [" + sfutil.green(self.shortname) + f"] {sfutil.warn('WARNING:')} No annotation file found, using whole slide.")

		self.shape = self.slide.dimensions
		self.filter_dimensions = self.slide.level_dimensions[-1]
		self.filter_magnification = self.filter_dimensions[0] / self.shape[0]
		goal_thumb_area = 4096*4096
		y_x_ratio = self.shape[1] / self.shape[0]
		thumb_x = sqrt(goal_thumb_area / y_x_ratio)
		thumb_y = thumb_x * y_x_ratio
		self.thumb = self.slide.get_thumbnail((int(thumb_x), int(thumb_y)))
		self.thumb_file = join(thumbs_path, f'{self.name}_thumb.jpg')
		imageio.imwrite(self.thumb_file, self.thumb)
		self.MPP = float(self.slide.properties[ops.PROPERTY_NAME_MPP_X])
		self.print("   * [" + sfutil.green(self.shortname) + f"] Microns per pixel: {self.MPP}")
		self.print("   * [" + sfutil.green(self.shortname) + f"] Loaded {filetype.upper()} of size {self.shape[0]} x {self.shape[1]}")

	def loaded_correctly(self):
		return bool(self.shape)

	def build_generator(self, size_px, size_um, stride_div, case_name, export=False, augment=False):
		shortname = sfutil._shortname(case_name)
		# Calculate window sizes, strides, and coordinates for windows
		tiles_path = join(self.export_folder, case_name)
		if not os.path.exists(tiles_path): os.makedirs(tiles_path)
		# Calculate pixel size of extraction window
		full_extract_px = int(size_um / self.MPP)
		downsample_desired = full_extract_px/size_px
		downsample_level = self.slide.get_best_level_for_downsample(downsample_desired)
		downsample_factor = self.slide.level_downsamples[downsample_level]
		downsample_shape = self.slide.level_dimensions[downsample_level]

		extract_px = int(full_extract_px / downsample_factor)
		stride = extract_px / stride_div #(should be int)

		self.print(f"   * [{sfutil.green(self.shortname)}] Extracting tiles of size {size_um}um, resizing from {extract_px}px -> {size_px}px ")
		if size_px > extract_px:
			self.print(f"   * [{sfutil.green(self.shortname)}] [{sfutil.fail('!WARN!')}]  Tiles will be up-scaled with cubic interpolation ({extract_px}px -> {size_px}px)")
		coord = []
		slide_x_size = downsample_shape[0] - extract_px
		slide_y_size = downsample_shape[1] - extract_px

		for y in np.arange(0, (downsample_shape[1]+1) - extract_px, stride):
			for x in np.arange(0, (downsample_shape[0]+1) - extract_px, stride):
				y = int(y)
				x = int(x)
				is_unique = ((y % extract_px == 0) and (x % extract_px == 0))
				coord.append([x, y, is_unique])

		# Load annotations as shapely.geometry objects
		ROI_SCALE = 10
		annPolys = [sg.Polygon(annotation.scaled_area(ROI_SCALE)) for annotation in self.annotations]
		roi_area = sum([poly.area for poly in annPolys])
		total_area = (self.shape[0]/ROI_SCALE) * (self.shape[1]/ROI_SCALE)

		roi_area_fraction = 1 if not total_area else (roi_area / total_area)

		total_logits_count = int(len(coord) * roi_area_fraction)
		# Create mask for indicating whether tile was extracted
		tile_mask = np.asarray([0 for i in range(len(coord))])
		self.tile_mask = None
		self.p_id = None if not self.pb else self.pb.add_bar(0, total_logits_count, endtext=sfutil.green(shortname))

		def generator():
			tile_counter=0
			for ci in range(len(coord)):
				c = coord[ci]
				filter_px = int(full_extract_px * self.filter_magnification)
				# Check if the center of the current window lies within any annotation; if not, skip
				x_coord = int((c[0]+extract_px/2)/ROI_SCALE)
				y_coord = int((c[1]+extract_px/2)/ROI_SCALE)
				if bool(annPolys) and not any([annPoly.contains(sg.Point(x_coord, y_coord)) for annPoly in annPolys]):
					continue
				tile_counter += 1
				if self.pb:
					self.pb.update(self.p_id, tile_counter)
				# Read the low-mag level for filter checking
				filter_region = np.asarray(self.slide.read_region(c, self.slide.level_count-1, [filter_px, filter_px]))[:,:,:-1]
				median_brightness = int(sum(np.median(filter_region, axis=(0, 1))))
				if median_brightness > 660:
					# Discard tile; median brightness (average RGB pixel) > 220
					continue
				if self.pb:
					self.pb.update_counter(1)
				# Read the region and discard the alpha pixels
				region = self.slide.read_region(c, downsample_level, [extract_px, extract_px])
				region = region.resize((size_px, size_px))
				region = region.convert('RGB')
				tile_mask[ci] = 1
				coord_label = ci
				unique_tile = c[2]
				if export and unique_tile:
					region.save(join(tiles_path, f'{shortname}_{ci}.jpg'), "JPEG")
					if augment:
						region.transpose(Image.ROTATE_90).save(join(tiles_path, f'{shortname}_{ci}_aug1.jpg'))
						region.transpose(Image.FLIP_TOP_BOTTOM).save(join(tiles_path, f'{shortname}_{ci}_aug2.jpg'))
						region.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM).save(join(tiles_path, f'{shortname}_{ci}_aug3.jpg'))
						region.transpose(Image.FLIP_LEFT_RIGHT).save(join(tiles_path, f'{shortname}_{ci}_aug4.jpg'))
						region.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT).save(join(tiles_path, f'{shortname}_{ci}_aug5.jpg'))
						region.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM).save(join(tiles_path, f'{shortname}_{ci}_aug6.jpg'))
						region.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM).save(join(tiles_path, f'{shortname}_{ci}_aug7.jpg'))
				yield region, coord_label, unique_tile
			if self.pb: 
				self.pb.end(self.p_id)
				self.print("   * [" + sfutil.green(self.shortname) + f"] {sfutil.info('Finished tile extraction')} ({sum(tile_mask)} tiles of {len(coord)} possible)")
			self.tile_mask = tile_mask

		return generator, slide_x_size, slide_y_size, stride, roi_area_fraction

	def load_csv_roi(self, path):
		roi_dict = {}
		with open(path, "r") as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			headers = next(reader, None)
			try:
				index_name = headers.index("ROI_Name")
				index_x = headers.index("X_base")
				index_y = headers.index("Y_base")
			except ValueError:
				raise IndexError('Unable to find "ROI_Name, "X_base", and "Y_base" columns in CSV file.')
			for row in reader:
				roi_name = row[index_name]
				x_coord = int(float(row[index_x]))
				y_coord = int(float(row[index_y]))
				
				if roi_name not in roi_dict:
					roi_dict.update({roi_name: ROIObject(roi_name)})
				roi_dict[roi_name].add_coord((x_coord, y_coord))

			for roi_object in roi_dict.values():
				self.annotations.append(roi_object)

			self.print("   * [" + sfutil.green(self.shortname) + f"] Number of ROIs: {len(self.annotations)}")

	def load_json_roi(self, path):
		with open(path, "r") as json_file:
			json_data = json.load(json_file)['shapes']
		for shape in json_data:
			area_reduced = np.multiply(shape['points'], JSON_ANNOTATION_SCALE)
			self.annotations.append(ROIObject(f"Object{len(self.annotations)}"))
			self.annotations[-1].add_shape(area_reduced)
		self.print("   * [" + sfutil.green(self.shortname) + "] Number of ROIs: {len(self.annotations)}")

class Convoluter:
	'''Class to guide the convolution/tessellation of tiles across a set of slides, within ROIs if provided. 
	Performs designated actions on tessellated tiles, which may include:
	
	 - image export (for generating a tile dataset, with or	without augmentation)
	 - logit predictions from saved Tensorflow model (logits may then be either saved or visualized with heatmaps)
	 - final layer weight calculation (saved into a CSV file)
	'''
	def __init__(self, size_px, size_um, num_classes, batch_size, use_fp16, save_folder='', roi_dir=None, augment=False):
		self.SLIDES = {}
		self.MODEL_DIR = None
		self.ROI_DIR = roi_dir
		self.SIZE_PX = size_px
		self.SIZE_UM = size_um
		self.NUM_CLASSES = num_classes
		self.BATCH_SIZE = batch_size
		self.DTYPE = tf.float16 if use_fp16 else tf.float32
		self.DTYPE_INT = tf.int16 if use_fp16 else tf.int32
		self.SAVE_FOLDER = save_folder
		self.STRIDE_DIV = 4
		self.MODEL_DIR = None
		self.AUGMENT = augment

		# BatchNormFix
		tf.keras.layers.BatchNormalization = sfutil.UpdatedBatchNormalization

	def load_slides(self, slides_array, category="None"):
		for slide_path in slides_array:
			name = slide_path.split('/')[-1][:-4]
			filetype = slide_path.split('/')[-1][-3:]
			self.SLIDES.update({name: { "name": name,
										"path": slide_path,
										"type": filetype,
										"category": category } })

	def convolute_slides(self, save_heatmaps=False, display_heatmaps=False, save_final_layer=False, export_tiles=True):
		'''Parent function to guide convolution across a whole-slide image and execute desired functions.

		Args:
			save_heatmaps: 				Bool, if true will save heatmap overlays as PNG files
			display_heatmaps:			Bool, if true will display interactive heatmap for each whole-slide image
			save_final_layer: 			Bool, if true will calculate and save final layer weights in CSV file
			export_tiles:				Bool, if true will save tessellated image tiles to subdirectory "tiles"

		Returns:
			None
		'''
		if not save_heatmaps and not display_heatmaps:
			# No need to calculate overlapping tiles
			print(f" + [{sfutil.info('INFO')}] Tessellating only non-overlapping tiles.")
			self.STRIDE_DIV = 1

		if export_tiles and not (display_heatmaps or save_final_layer or save_heatmaps):
			print(f" + [{sfutil.info('INFO')}] Exporting tiles only, no new calculations or heatmaps will be generated.")
			pb = progress_bar.ProgressBar(bar_length=5, counter_text='tiles')
			pool = Pool(NUM_THREADS)
			pool.map(lambda slide: self.export_tiles(self.SLIDES[slide], pb), self.SLIDES)

		elif not display_heatmaps:

			# Create a CSV writing queue to prevent conflicts with multithreadings
			q = Queue()
			self.queue_open = True

			def map_logits_calc(case_name, pb):
				slide = self.SLIDES[case_name]
				shortname = sfutil._shortname(case_name)
				category = slide['category']
				pb.print(f" + Working on case {sfutil.green(shortname)}")
				logits, final_layer, final_layer_labels, logits_flat = self.calculate_logits(slide, export_tiles, save_final_layer, pb=pb)
				if save_heatmaps:
					self.gen_heatmaps(slide, logits, self.SIZE_PX, case_name, save=True)
				if save_final_layer:
					# Add CSV writing to the queue
					q.put([final_layer, final_layer_labels, logits_flat, case_name, category])

			case_names = self.SLIDES.keys()
			pb = progress_bar.ProgressBar(bar_length=5, counter_text='tiles')
			pool = ThreadPool(4)

			def close_queue(r):
				self.queue_open = False

			# Create a thread to coordinate multithreading of logits calculation
			pool.map_async(lambda case_name: map_logits_calc(case_name, pb), case_names, callback=close_queue)
			
			# Use the main thread to make the heatmaps
			while self.queue_open:
				final_layer, final_layer_labels, logits_flat, case_name, category = q.get()
				self.export_weights(final_layer, final_layer_labels, logits_flat, case_name, category)
				q.task_done()

		else:
			for case_name in self.SLIDES:
				slide = self.SLIDES[case_name]
				shortname = sfutil._shortname(case_name)
				category = slide['category']
				print(f" + Working on case {shortname} ({category})")

				logits, final_layer, final_layer_labels, logits_flat = self.calculate_logits(slide, export_tiles, save_final_layer)
				if save_heatmaps:
					self.gen_heatmaps(slide, logits, self.SIZE_PX, case_name, save=True)
				if save_final_layer:
					self.export_weights(final_layer, final_layer_labels, logits_flat, case_name, category)
				if display_heatmaps:
					self.gen_heatmaps(slide, logits, self.SIZE_PX, case_name, save=False)

	def export_tiles(self, slide, pb):
		case_name = slide['name']
		category = slide['category']
		path = slide['path']
		filetype = slide['type']
		shortname = sfutil._shortname(case_name)

		pb.print(f" + Exporting tiles for case {sfutil.green(shortname)}")

		whole_slide = SlideReader(path, filetype, self.SAVE_FOLDER, self.ROI_DIR, pb=pb)
		if not whole_slide.loaded_correctly(): return
		gen_slice, _, _, _, _ = whole_slide.build_generator(self.SIZE_PX, self.SIZE_UM, self.STRIDE_DIV, case_name, 
															export=True, 
															augment=self.AUGMENT)
		for tile, coord, unique in gen_slice(): 
			pass

	def _parse_function(self, image, label, mask):
		parsed_image = tf.image.per_image_standardization(image)
		parsed_image = tf.image.convert_image_dtype(parsed_image, self.DTYPE)
		return parsed_image, label, mask

	def build_model(self, model_dir, SFM=None):
		self.MODEL_DIR = model_dir
		self.SFM = SFM
		_model = tf.keras.models.load_model(self.MODEL_DIR)
		self.model = tf.keras.models.Model(inputs=[_model.input, _model.layers[0].layers[0].input],
										   outputs=[_model.layers[0].layers[-1].output, _model.layers[1].output])

	def calculate_logits(self, slide, export_tiles=False, final_layer=False, pb=None):
		'''Returns logits and final layer weights'''
		warnings.simplefilter('ignore', Image.DecompressionBombWarning)
		case_name = slide['name']
		path = slide['path']
		filetype = slide['type']

		# Load whole-slide-image into Numpy array
		whole_slide = SlideReader(path, filetype, self.SAVE_FOLDER, self.ROI_DIR, pb=pb)

		# Create tile coordinate generator
		gen_slice, x_size, y_size, stride_px, roi_area_fraction = whole_slide.build_generator(self.SIZE_PX, self.SIZE_UM, self.STRIDE_DIV, case_name, 
																		 export=export_tiles)

		# Generate dataset from coordinates
		with tf.name_scope('dataset_input'):
			tile_dataset = tf.data.Dataset.from_generator(gen_slice, (tf.uint8, tf.int64, tf.bool))
			tile_dataset = tile_dataset.map(self._parse_function, num_parallel_calls=8)
			tile_dataset = tile_dataset.batch(self.BATCH_SIZE, drop_remainder=False)

		logits_arr = []
		labels_arr = []
		x_logits_len = int(x_size / stride_px) + 1
		y_logits_len = int(y_size / stride_px) + 1
		total_logits_count = int((x_logits_len * y_logits_len) * roi_area_fraction)

		count = 0
		prelogits_arr = [] # Final layer weights 
		logits_arr = []	# Logits (predictions) 
		unique_arr = []	# Boolean array indicating whether tile is unique (non-overlapping) 

		# Iterate through generator to calculate logits +/- final layer weights for all tiles
		for batch_images, batch_labels, batch_unique in tile_dataset:
			count = min(count, total_logits_count)
			prelogits, logits = self.model.predict([batch_images, batch_images])
			if not pb:
				progress_bar.bar(count, total_logits_count, text = "Calculated {} images out of {}. "
																	.format(min(count, total_logits_count),
																		total_logits_count))
			count += len(batch_images)
			#new_prelogits, new_logits, new_labels, new_unique
			prelogits_arr = prelogits if prelogits_arr == [] else np.concatenate([prelogits_arr, prelogits])
			logits_arr = logits if logits_arr == [] else np.concatenate([logits_arr, logits])
			labels_arr = batch_labels if labels_arr == [] else np.concatenate([labels_arr, batch_labels])
			unique_arr = batch_unique if unique_arr == [] else np.concatenate([unique_arr, batch_unique])
		progress_bar.end()

		# Sort the output (may be shuffled due to multithreading)
		try:
			sorted_indices = labels_arr.argsort()
		except AttributeError:
			# This occurs when the list is empty, likely due to an empty annotation area
			raise AttributeError("No tile calculations performed for this image, are you sure the annotation area isn't empty?")
		logits_arr = logits_arr[sorted_indices]
		labels_arr = labels_arr[sorted_indices]
		
		# Perform same functions on final layer weights
		flat_unique_logits = None
		if final_layer:
			prelogits_arr = prelogits_arr[sorted_indices]
			unique_arr = unique_arr[sorted_indices]
			# Find logits from non-overlapping tiles (will be used for metadata for saved final layer weights CSV)
			flat_unique_logits = [logits_arr[l] for l in range(len(logits_arr)) if unique_arr[l]]
			prelogits_out = [prelogits_arr[p] for p in range(len(prelogits_arr)) if unique_arr[p]]
			prelogits_labels = [labels_arr[l] for l in range(len(labels_arr)) if unique_arr[l]]
		else:
			prelogits_out = None
			prelogits_labels = None

		# Expand logits back to a full 2D map spanning the whole slide,
		#  supplying values of "0" where tiles were skipped by the tile generator
		expanded_logits = [[0] * self.NUM_CLASSES] * len(whole_slide.tile_mask)
		li = 0
		for i in range(len(expanded_logits)):
			if whole_slide.tile_mask[i] == 1:
				expanded_logits[i] = logits_arr[li]
				li += 1
		expanded_logits = np.asarray(expanded_logits, dtype=float)
		expanded_logits_message = f"   * Expanded_logits size: {expanded_logits.shape}; resizing to y:{y_logits_len} and x:{x_logits_len}"
		if not pb:
			print(expanded_logits_message)
		else:
			pb.print(expanded_logits_message)		

		# Resize logits array into a two-dimensional array for heatmap display
		logits_out = np.resize(expanded_logits, [y_logits_len, x_logits_len, self.NUM_CLASSES])

		return logits_out, prelogits_out, prelogits_labels, flat_unique_logits

	def export_weights(self, output, labels, logits, name, category):
		'''Exports final layer weights (and logits) for non-overlapping tiles into a CSV file.'''
		print(" + Writing csv...")
		csv_started = os.path.exists(join(self.SAVE_FOLDER, 'final_layer_weights.csv'))
		write_mode = 'a' if csv_started else 'w'
		with open(join(self.SAVE_FOLDER, 'final_layer_weights.csv'), write_mode) as csv_file:
			csv_writer = csv.writer(csv_file, delimiter = ',')
			if not csv_started:
				csv_writer.writerow(["Tile_num", "Case", "Category"] + [f"Logits{l}" for l in range(len(logits[0]))] + [f"Node{n}" for n in range(len(output[0]))])
			for l in range(len(output)):
				logit = logits[l].tolist()
				out = output[l].tolist()
				csv_writer.writerow([labels[l], name, category] + logit + out)

	def gen_heatmaps(self, slide, logits, size, name, save=True):
		'''Displays and/or saves logits as a heatmap overlay.'''
		#print(" + Received logits, size=%s, (%s x %s)" % (size, len(logits), len(logits[0])))
		#print(" + Calculating overlay matrix and displaying with dynamic resampling...")
		image_file = slide['path']
		filetype = slide['type']
		fig = plt.figure(figsize=(18, 16))
		ax = fig.add_subplot(111)
		fig.subplots_adjust(bottom = 0.25, top=0.95)

		if image_file[-4:] == ".svs":
			whole_slide = SlideReader(image_file, filetype, self.SAVE_FOLDER, self.ROI_DIR)
			im = whole_slide.thumb #plt.imread(whole_svs.thumb)
		else:
			im = plt.imread(image_file)

		implot = ax.imshow(im, zorder=0) if save else FastImshow(im, ax, extent=None, tgt_res=1024)
		im_extent = implot.get_extent() if save else implot.extent
		#extent = [im_extent[0] + size/2, im_extent[1] - size/2, im_extent[2] - size/2, im_extent[3] + size/2]
		extent = im_extent

		gca = plt.gca()
		gca.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

		# Define color map
		jetMap = np.linspace(0.45, 0.95, 255)
		cmMap = cm.nipy_spectral(jetMap)
		newMap = mcol.ListedColormap(cmMap)

		heatmap_dict = {}

		def slider_func(val):
			for h, s in heatmap_dict.values():
				h.set_alpha(s.val)

		# Make heatmaps and sliders
		for i in range(self.NUM_CLASSES):
			heatmap = ax.imshow(logits[:, :, i], extent=extent, cmap=newMap, alpha = 0.0, interpolation='none', zorder=10) #bicubic
			if save:
				heatmap_dict.update({i: heatmap})
			else:
				ax_slider = fig.add_axes([0.25, 0.2-(0.2/self.NUM_CLASSES)*i, 0.5, 0.03], facecolor='lightgoldenrodyellow')
				slider = Slider(ax_slider, f'Class {i}', 0, 1, valinit = 0)
				heatmap_dict.update({f"Class{i}": [heatmap, slider]})
				slider.on_changed(slider_func)

		# Save of display heatmap overlays
		if save:
			mp.savefig(os.path.join(self.SAVE_FOLDER, f'{name}-raw.png'), bbox_inches='tight')
			for i in range(self.NUM_CLASSES):
				heatmap_dict[i].set_alpha(0.6)
				mp.savefig(os.path.join(self.SAVE_FOLDER, f'{name}-{i}.png'), bbox_inches='tight')
				heatmap_dict[i].set_alpha(0.0)
			mp.close()
		else:
			fig.canvas.set_window_title(name)
			implot.show()
			plt.show()

def get_args():
	parser = argparse.ArgumentParser(description = 'Convolutionally applies a saved Tensorflow model to a larger image, displaying the result as a heatmap overlay.')
	parser.add_argument('-m', '--model', help='Path to Tensorflow model directory containing stored checkpoint.')
	parser.add_argument('-s', '--slide', help='Path to whole-slide image (SVS or JPG format) or folder of images (SVS or JPG) to analyze.')
	parser.add_argument('-o', '--out', help='Path to directory in which exported images and data will be saved.')
	parser.add_argument('-c', '--classes', type=int, default = 1, help='Number of unique output classes contained in the model.')
	parser.add_argument('-b', '--batch', type=int, default = 64, help='Batch size for which to run the analysis.')
	parser.add_argument('--px', type=int, default=512, help='Size of image patches to analyze, in pixels.')
	parser.add_argument('--um', type=float, default=255.3856, help='Size of image patches to analyze, in microns.')
	parser.add_argument('--fp16', action="store_true", help='Use Float16 operators (half-precision) instead of Float32.')
	parser.add_argument('--save', action="store_true", help='Save heatmaps to PNG file instead of displaying.')
	parser.add_argument('--final', action="store_true", help='Calculate and export image tiles and final layer weights.')
	parser.add_argument('--display', action="store_true", help='Display results with interactive heatmap for each whole-slide image.')
	parser.add_argument('--export', action="store_true", help='Save extracted image tiles.')
	parser.add_argument('--augment', action="store_true", help='Augment extracted tiles with flipping/rotating.')
	parser.add_argument('--num_threads', type=int, help='Number of threads to use when tessellating.')
	return parser.parse_args()

if __name__==('__main__'):
	# Disable warnings to maintain clean output
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	#tf.logging.set_verbosity(tf.logging.ERROR)
	args = get_args()

	if not args.out: args.out = args.slide
	if args.num_threads: NUM_THREADS = args.num_threads

	c = Convoluter(args.px, args.um, args.classes, args.batch, args.fp16, args.out, augment=args.augment)

	# Load images/slides
	# If a single file is provided with the --slide flag, then load only that image
	if isfile(args.slide):
		c.load_slides(args.slide)
	else:
		# Otherwise, assume the --slide flag provided a directory and attempt to load images in the directory 
		# First, load all images in the directory, without assigning any category labels
		slide_list = [join(args.slide, i) for i in os.listdir(args.slide) if (isfile(join(args.slide, i)) and (i[-3:].lower() in ("svs", "jpg")))]	
		c.load_slides(slide_list)
		# Next, load images in subdirectories, assigning category labels by subdirectory name
		dir_list = [d for d in os.listdir(args.slide) if not isfile(join(args.slide, d))]
		for directory in dir_list:
			# Ignore images if in the thumbnails or QuPath project directory
			if directory in ["thumbs", "QuPath_Project"]: continue
			slide_list = [join(args.slide, directory, i) for i in os.listdir(join(args.slide, directory)) if (isfile(join(args.slide, directory, i)) and (i[-3:].lower() in ("svs", "jpg")))]	
			c.load_slides(slide_list, category=directory)
			
	c.build_model(args.model)
	c.convolute_slides(args.save, args.display, args.final, args.export)