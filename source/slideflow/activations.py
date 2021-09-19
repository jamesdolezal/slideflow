import sys
import os
import shutil
import csv
import pickle
import time
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcol
import seaborn as sns
import scipy.stats as stats
import slideflow.util as sfutil
import slideflow.io as sfio
import shapely.geometry as sg

from slideflow.util import log, ProgressBar, TCGA, StainNormalizer
from slideflow.util.fastim import FastImshow
from slideflow.model import ModelActivationsInterface
from sklearn.linear_model import LogisticRegression
from os.path import join, exists
from statistics import mean
from math import isnan
from copy import deepcopy
from matplotlib.widgets import Slider
from functools import partial
from multiprocessing.dummy import Process as DProcess
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from tqdm import tqdm

# TODO: change slide_node_dict and slide_logits_dict to be multidimensional arrays
#       rather than this nested dictionary garbage

class ActivationsError(Exception):
    pass

class ActivationsVisualizer:
    '''Loads annotations, saved layer activations, and prepares output saving directories.
        Will also read/write processed activations to a PKL cache file to save time in future iterations.'''

    def __init__(self,
                 model,
                 tfrecords,
                 export_dir,
                 image_size,
                 annotations=None,
                 outcome_label_headers=None,
                 focus_nodes=None,
                 normalizer=None,
                 normalizer_source=None,
                 cache=None,
                 batch_size=32,
                 activations_export=None,
                 max_tiles_per_slide=0,
                 min_tiles_per_slide=0,
                 manifest=None,
                 layers='postconv',
                 include_logits=True):

        '''Calculates activations from model.

        Args:
            model:					Path to model from which to calculate activations
            tfrecords:				List of tfrecords paths
            export_dir:				Export directory in which to save cache files and output files
            image_size:				Int, width/height of input images in pixels
            annotations:			Path to CSV file containing slide annotations
            outcome_label_headers:	String, name of outcome header in annotations file,
                                        used to compare activations between categories
            focus_nodes:			List of int, nodes on which to focus when generating cross-category statistics
            normalizer:				String, which real-time normalization to use on images taken from TFRecords
            normalizer_source:		String, path to image to use as source for real-time normalization
            cache:		            File in which to store activations PKL cache
            batch_size:				Batch size to use during activations calculations
            activations_export:		Filename for CSV export of activations
            max_tiles_per_slide:	Maximum number of tiles from which to generate activations for each slide
            min_tiles_per_slide:	If provided, will only evaluate slides with a given minimum number of tiles.
            manifest:				Optional, dict mapping tfrecords to number of tiles contained.
                                        Used for progress bars and min_tiles_per_slide.
            layers:                 Names of layers from the model from which to calculate activations.
            include_logits:         Bool. If true, will also calculate and store logits.
        '''
        self.categories = []
        self.used_categories = []
        self.slide_category_dict = {}
        self.slide_node_dict = {}
        self.slide_logits_dict = {}
        self.slide_loc_dict = {}
        self.node_cat_dict = {}
        self.num_features = None
        self.focus_nodes = focus_nodes
        self.sorted_nodes = None
        self.manifest = manifest
        self.model = model
        self.tfrecords = np.array(tfrecords)
        self.slides = sorted([sfutil.path_to_name(tfr) for tfr in self.tfrecords])
        self.export_dir = export_dir
        self.image_size = image_size

        if not isinstance(layers, list): layers = [layers]

        if min_tiles_per_slide and not manifest:
            raise ActivationsError("'manifest' must be provided if specifying min_tiles_per_slide.")
        if min_tiles_per_slide:
            self._filter_by_min_tiles(min_tiles_per_slide)

        if cache=='default':
            cache = join(export_dir, 'activations_cache.pkl')
        elif cache:
            cache = join(export_dir, cache)

        if not exists(export_dir):
            os.makedirs(export_dir)

        # Load annotations if provided
        if annotations and outcome_label_headers:
            self.load_annotations(annotations, outcome_label_headers)

        # Load activations
        # Load from PKL (cache) if present
        if cache and exists(cache):
            # Load saved PKL cache
            log.info(f'Loading pre-calculated predictions and activations from {cache}...')
            with open(cache, 'rb') as pt_pkl_file:

                self.slide_node_dict, self.slide_logits_dict, self.slide_loc_dict = pickle.load(pt_pkl_file)
                first_slide = list(self.slide_node_dict.keys())[0]
                logits = 	  list(self.slide_logits_dict[first_slide].keys())
                self.nodes =  list(self.slide_node_dict[first_slide].keys())
                if max_tiles_per_slide:
                    log.info(f'Filtering activations to maximum {max_tiles_per_slide} tiles per slide')
                    for slide in self.slide_node_dict.keys():
                        for n in self.nodes:
                            if len(self.slide_node_dict[slide][n]) > max_tiles_per_slide:
                                self.slide_node_dict[slide][n] = self.slide_node_dict[slide][n][:max_tiles_per_slide]
                        for l in logits:
                            if len(self.slide_logits_dict[slide][l]) > max_tiles_per_slide:
                                self.slide_logits_dict[slide][l] = self.slide_logits_dict[slide][l][:max_tiles_per_slide]
                        if len(self.slide_loc_dict[slide]) > max_tiles_per_slide:
                            self.slide_loc_dict[slide] = self.slide_loc_dict[slide][:max_tiles_per_slide]

        # Otherwise will need to generate new activations from a given model
        else:
            self.generate_activations_from_model(model,
                                                 batch_size=batch_size,
                                                 export=activations_export,
                                                 normalizer=normalizer,
                                                 normalizer_source=normalizer_source,
                                                 layers=layers,
                                                 include_logits=include_logits,
                                                 max_tiles_per_slide=max_tiles_per_slide,
                                                 cache=cache)

            if self.slide_node_dict != {}:
                self.nodes = list(self.slide_node_dict[list(self.slide_node_dict.keys())[0]].keys())
            else:
                self.nodes = []

        # Now delete slides not included in our filtered TFRecord list
        loaded_slides = list(self.slide_node_dict.keys())
        for loaded_slide in loaded_slides:
            if loaded_slide not in self.slides:
                self.remove_slide(loaded_slide)

        # Now screen for missing slides in activations
        missing_slides = []
        for slide in self.slides:
            if slide not in self.slide_node_dict:
                missing_slides += [slide]
            elif self.slide_node_dict[slide][0] == []:
                missing_slides += [slide]
        num_loaded = len(self.slides)-len(missing_slides)
        log.info(f'Loaded activations from {num_loaded}/{len(self.slides)} slides ({len(missing_slides)} missing)')
        if missing_slides:
            log.warning(f'Activations missing for {len(missing_slides)} slides')

        # Record which categories have been included in the specified tfrecords
        if self.categories:
            self.used_categories = list(set([self.slide_category_dict[slide] for slide in self.slides]))
            self.used_categories.sort()
        log.info(f'Observed categories (total: {len(self.used_categories)}):')
        for c in self.used_categories:
            log.info(f'\t{c}')
        # Show total number of features
        if self.num_features is None:
            self.num_features = len(list(self.slide_node_dict[self.slides[0]].keys()))
        log.info(f'Number of activation features: {self.num_features}')

    def _save_node_statistics_to_csv(self, sorted_nodes, slide_node_dict, filename, tile_stats=None, slide_stats=None):
        '''Internal function to exports statistics (ANOVA p-values and slide-level averages) to CSV.

        Args:
            sorted_nodes:		List of node IDs (int) sorted in order of significance.
            slide_node_dict:	Dict mapping slides to dict of nodes mapping to slide-level values.
                                    Slide-level node values could be mean, median, thresholded, or other.
            filename:			Filename
            tile_stats:			Dictionary mapping nodes to a dict of tile-level stats
                                    containing 'p' (ANOVA P-value) and 'f' (ANOVA F-value) for each node
                                    As calculated elsewhere by comparing node activations between categories
            slide_stats:		Dictionary mapping nodes to a dict of slide-level stats
                                    containing 'p' (ANOVA P-value), 'f' (ANOVA F-value),
                                    and 'num_above_threshold' for each node,
                                    as calculated elsewhere by comparing node activations between categories
        '''
        # Save results to CSV
        log.info(f'Writing results to {sfutil.green(filename)}...')
        with open(filename, 'w') as outfile:
            csv_writer = csv.writer(outfile)
            header = ['slide', 'category'] + [f'FLNode{n}' for n in sorted_nodes]
            csv_writer.writerow(header)
            for slide in self.slides:
                category = self.slide_category_dict[slide]
                row = [slide, category] + [slide_node_dict[slide][n] for n in sorted_nodes]
                csv_writer.writerow(row)
            if tile_stats:
                csv_writer.writerow(['Tile statistic', 'ANOVA P-value'] + [tile_stats[n]['p'] for n in sorted_nodes])
                csv_writer.writerow(['Tile statistic', 'ANOVA F-value'] + [tile_stats[n]['f'] for n in sorted_nodes])
            if slide_stats:
                csv_writer.writerow(['Slide statistic', 'ANOVA P-value'] + [slide_stats[n]['p'] for n in sorted_nodes])
                csv_writer.writerow(['Slide statistic', 'ANOVA F-value'] + [slide_stats[n]['f'] for n in sorted_nodes])

    def _filter_by_min_tiles(self, min_tiles):
        unique_slides = []
        included_tfrecords = []
        for tfr in self.tfrecords:
            num_tiles = self.manifest[tfr]['total']
            if num_tiles < min_tiles:
                log.info(f'Skipped {sfutil.green(sfutil.path_to_name(tfr))} (has {num_tiles} tiles, min {min_tiles})')
            else:
                unique_slides += [sfutil.path_to_name(tfr)]
                included_tfrecords += [tfr]
        unique_slides = list(set(unique_slides))
        self.tfrecords = included_tfrecords
        self.slides = sorted(unique_slides)
        log.info(f'Total slides after filtering by minimum number of tiles: {len(self.slides)}')

    def slide_tile_dict(self):
        '''Generates dictionary mapping slides to list of node activations for each tile.

        Example (3 nodes):
            { 'Slide1': [[0.1, 0.2, 0.4], # Slide1, node activations for tile1
                         [0.5, 0.1, 0.7], # Slide1, node activations for tile2
                         [0.6, 0.9, 0.1]] # Slide1, node activations for tile3
            }
        '''
        result = {}
        for slide in self.slides:
            num_tiles = len(self.slide_node_dict[slide][0])
            result.update({slide: [[self.slide_node_dict[slide][node][tile_index] for node in self.nodes]
                                                                                  for tile_index in range(num_tiles)]})
        return result

    def map_to_predictions(self, x=0, y=0):
        '''Returns coordinates and metadata for tile-level predictions for all tiles,
        which can be used to create a TFRecordMap.

        Args:
            x:			Int, identifies the outcome category id for which predictions will be mapped to the X-axis
            y:			Int, identifies the outcome category id for which predictions will be mapped to the Y-axis

        Returns:
            mapped_x:	List of x-axis coordinates (predictions for the category 'x')
            mapped_y:	List of y-axis coordinates (predictions for the category 'y')
            umap_meta:	List of dictionaries containing tile-level metadata (used for TFRecordMap)
        '''
        umap_x, umap_y, umap_meta = [], [], []
        for slide in self.slides:
            for tile_index in range(len(self.slide_logits_dict[slide][0])):
                umap_x += [self.slide_logits_dict[slide][x][tile_index]]
                umap_y += [self.slide_logits_dict[slide][y][tile_index]]
                umap_meta += [{
                    'slide': slide,
                    'index': tile_index
                }]
        return np.array(umap_x), np.array(umap_y), umap_meta

    def get_activations(self):
        '''Returns dictionary mapping slides to tile-level node activations.

        Example (3 nodes):
            { 'Slide1': [[0.1, 0.5, 0.6], # Slide1, node1 activations for all tiles
                         [0.2, 0.1, 0.7], # Slide1, node2 activations for all tiles
                         [0.4, 0.7, 0.1]] # Slide1, node3 activations for all tiles
            }
        '''
        return self.slide_node_dict

    def get_predictions(self):
        '''Returns dictionary mapping slides to tile-level logit predictions.

        Example (2 outcome categories):
            { 'Slide1': [[0.1, 0.9, 0.6], # Slide1, logit predictions for category 1 for all tiles
                         [0.9, 0.1, 0.4], # Slide1, logit predictions for category 2 for all tiles
            }
        '''
        return self.slide_logits_dict

    def get_slide_level_linear_predictions(self):
        '''Returns slide-level predictions assuming the model is predicting a linear outcome.

        Returns:
            dict:		Dictionary mapping slide names to final slide-level predictions
                            for each outcome cateogry, calculated as the average predicted value
                            in the outcome category for all tiles in the slide.
                            Example:
                                { 'slide1': {
                                    0: 0.24,	# Outcome category 0
                                    1: 0.15,	# Outcome category 1
                                    2: 0.61 }}	# Outcome category 2
        '''
        first_slide = list(self.slide_logits_dict.keys())[0]
        outcome_labels = sorted(list(self.slide_logits_dict[first_slide].keys()))
        slide_predictions = {slide: {o: mean(self.slide_logits_dict[slide][o]) for o in outcome_labels}
                                                                               for slide in self.slide_logits_dict}
        return slide_predictions

    def get_slide_level_categorical_predictions(self, prediction_filter=None):
        '''Returns slide-level predictions assuming the model is predicting a categorical outcome.

        Args:
            prediction_filter:	(optional) List of int. If provided, will restrict predictions to only these
                                    categories, with final prediction being based based on highest logit
                                    among these categories.

        Returns:
            slide_predictions:	Dictionary mapping slide names to final slide-level predictions.
            slide_percentages:	This is a dictionary mapping slide names to a dictionary for each category,
                                    which maps the category id to the percent of tiles in the slide
                                    predicted to be this category.
                                    Example:
                                        { 'slide1': {
                                            0: 0.24,
                                            1: 0.15,
                                            2: 0.61 }}
                                If linear model, this is the same as slide_predictions.
        '''
        slide_predictions = {}
        slide_percentages = {}
        first_slide = list(self.slide_logits_dict.keys())[0]
        outcome_labels = sorted(list(self.slide_logits_dict[first_slide].keys()))
        for slide in self.slide_logits_dict:
            num_tiles = len(self.slide_logits_dict[slide][0])
            tile_predictions = []
            for i in range(num_tiles):
                calculated_logits = [self.slide_logits_dict[slide][o][i] for o in outcome_labels]
                if prediction_filter:
                    filtered_calculated_logits = [calculated_logits[o] for o in prediction_filter]
                else:
                    filtered_calculated_logits = calculated_logits
                tile_predictions += [calculated_logits.index(max(filtered_calculated_logits))]
            slide_prediction_values = {o: (tile_predictions.count(o)/len(tile_predictions)) for o in outcome_labels}
            slide_percentages.update({slide: slide_prediction_values})
            slide_predictions.update({slide: max(slide_prediction_values, key=lambda l: slide_prediction_values[l])})
        return slide_predictions, slide_percentages

    def load_annotations(self, annotations, outcome_label_headers):
        '''Loads annotations from a given file with the specified outcome header.

        Args:
            annotations:				Path to CSV annotations file.
            outcome_label_headers:		String, name of column header from which to read outcome variables.
        '''
        with open(annotations, 'r') as ann_file:
            log.info('Reading annotations...')
            ann_reader = csv.reader(ann_file)
            header = next(ann_reader)
            slide_i = header.index(TCGA.slide)
            category_i = header.index(outcome_label_headers)
            for row in ann_reader:
                slide = row[slide_i]
                category = row[category_i]
                if slide not in self.slides: continue
                self.slide_category_dict.update({slide:category})

        self.categories = list(set(self.slide_category_dict.values()))

        if self.slide_node_dict:
            # If initial loading has been completed already, make note of observed categories in given header
            for slide in self.slides:
                try:
                    if self.slide_node_dict[slide][0] != []:
                        self.used_categories = list(set(self.used_categories + [self.slide_category_dict[slide]]))
                        self.used_categories.sort()
                except KeyError:
                    # Skip unknown slide
                    pass
            log.info(f'Observed categories (total: {len(self.used_categories)}):')
            for c in self.used_categories:
                log.info(f'\t{c}')

    def get_tile_node_activations_by_category(self, node):
        '''For each outcome category, calculates activations of a given node across all tiles in the category.
        Requires annotations to have been loaded with load_annotations()

        Args:
            node:		Int, id of node.

        Returns:
            List of node activations separated by category.
                Example:
                [[0.1, 0.2, 0.7, 0.1, 0.0], # Activations for node 'N' across all tiles from slides in category 1
                 [0.8, 0.2, 0.1]] 			# Activations for node 'N' across all tiles from slides in category 2
        '''
        if not self.categories:
            log.warning('Unable to calculate activations by category. Please load annotations with load_annotations()')
            return
        tile_node_activations_by_category = []
        for c in self.used_categories:
            nodelist = [self.slide_node_dict[pt][node] for pt in self.slides if self.slide_category_dict[pt] == c]
            tile_node_activations_by_category += [[nodeval for nl in nodelist for nodeval in nl]]
        return tile_node_activations_by_category

    def get_top_nodes_by_slide(self):
        '''First, slide-level average node activations are calculated for all slides.
            Then, the significance of the difference in average node activations between for slides
            belonging to the different outcome categories is calculated using ANOVA.
            This function then returns a list of all nodes, sorted by ANOVA p-value (most significant first).
        '''
        # First ensure basic stats have been calculated
        if not hasattr(self, 'sorted_nodes'):
            self.calculate_activation_averages_and_stats()

        return self.sorted_nodes

    def get_top_nodes_by_tile(self):
        '''First, tile-level average node activations are calculated for all tiles.
            Then, the significance of the difference in node activations for tiles
            belonging to the different outcome categories is calculated using ANOVA.
            This function then returns a list of all nodes, sorted by ANOVA p-value (most significant first).
        '''
        # First ensure basic stats have been calculated
        if not hasattr(self, 'sorted_nodes'):
            self.calculate_activation_averages_and_stats()

        return self.nodes

    def find_neighbors(self, neighbor_AV, neighbor_slides, n_neighbors=5, algorithm='ball_tree'):
        '''Finds neighboring tiles for a given ActivationsVisualizer and list of slides.

        Args:
            neighbor_AV:		ActivationsVisualizer, will be used to look for neighbors
            neighbor_slides:	Either a single slide name or a list of slide names.
                                    Corresponds to slides in the provided neighboring AV.
                                    Will look for neighbors to all tiles in these slides.
            n_neighbors:		Number of neighbors to find for each tile
            algorithm:			NearestNeighbors algorithm, either 'kd_tree', 'ball_tree', or 'brute'

        Returns:
            Dict mapping slide names (from self.slides) to tile indices for tiles
                that were found to be neighbors to the provided neighbor_AV and neighbor_slides.
        '''
        if not isinstance(neighbor_slides, list): neighbor_slides = [neighbor_slides]

        # Setup source slide-tile indices
        slide_tile_indices = []
        for slide in self.slide_node_dict:
            for tile_index in range(len(self.slide_node_dict[slide][0])):
                slide_tile_indices += [(slide, tile_index)]

        # Setup neighbor slide-tile indices
        neighbor_slide_tile_indices = []
        for slide in neighbor_slides:
            if slide not in neighbor_AV.slide_node_dict:
                raise ActivationsError(f'Slide {slide} not found in neighboring ActivationsVisualizer')
            for tile_index in range(len(neighbor_AV.slide_node_dict[slide][0])):
                neighbor_slide_tile_indices += [(slide, tile_index)]

        log.info('Initializing nearest neighbor search...')

        X = np.array([[self.slide_node_dict[slide][n][tile_index] for n in self.nodes]
                      for (slide, tile_index) in slide_tile_indices])

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, n_jobs=-1).fit(X)
        neighbors = {}

        log.info('Searching for nearest neighbors...')
        neighbor_activations = [[neighbor_AV.slide_node_dict[slide][n][tile_index] for n in neighbor_AV.nodes]
                                for (slide, tile_index) in neighbor_slide_tile_indices]

        _, all_indices = nbrs.kneighbors(neighbor_activations)

        for indices_by_tile in all_indices:
            for index in indices_by_tile:
                neighboring_slide, tile_index = slide_tile_indices[index]
                if neighboring_slide in neighbors and tile_index not in neighbors[neighboring_slide]:
                    neighbors[neighboring_slide] += [tile_index]
                else:
                    neighbors.update({neighboring_slide: [tile_index]})
        return neighbors

    def generate_activations_from_model(self,
                                        model,
                                        batch_size=16,
                                        export=None,
                                        normalizer=None,
                                        normalizer_source=None,
                                        layers='postconv',
                                        include_logits=True,
                                        max_tiles_per_slide=0,
                                        cache=None):

        '''Calculates activations from a given model.

        Args:
            model:		Path to Tensorflow model from which to calculate final layer activations.
            batch_size:	Batch size for model predictions.
            export:		String (default: None). If provided, will export CSV of activations with this filename.'''

        # Rename tfrecord_array to tfrecords
        log.info(f'Calculating activations from {sfutil.green(model)}, max {max_tiles_per_slide} tiles per slide.')
        if not isinstance(layers, list): layers = [layers]

        # Load model
        combined_model = ModelActivationsInterface(model, layers=layers, include_logits=include_logits)
        unique_slides = list(set([sfutil.path_to_name(tfr) for tfr in self.tfrecords]))
        self.num_features = combined_model.num_features

        # Prepare normalizer
        if normalizer: log.info(f'Using realtime {normalizer} normalization')
        normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

        # Prepare PKL export dictionary
        for slide in unique_slides:
            if slide not in self.slide_node_dict:
                self.slide_node_dict.update({slide: {}})
            if slide not in self.slide_logits_dict:
                self.slide_logits_dict.update({slide: {}})
            if slide not in self.slide_loc_dict:
                self.slide_loc_dict.update({slide: []})

        # Calculate final layer activations for each tfrecord
        fla_start_time = time.time()
        nodes_names, logits_names = [], []
        detected_logit_structure = False
        include_tfrecord_loc = True
        slides_to_remove = []
        if export:
            outfile = open(export, 'w')
            csvwriter = csv.writer(outfile)

        for t, tfrecord in enumerate(self.tfrecords):
            dataset = tf.data.TFRecordDataset(tfrecord)
            tfr_features, tfr_img_type = sfio.tfrecords.detect_tfrecord_format(tfrecord)
            if tfr_features is None:
                log.warning(f"Unable to read tfrecord at {tfrecord} - is it empty?")
                slides_to_remove += [sfutil.path_to_name(tfrecord)]
                continue
            if 'loc_x' not in tfr_features:
                include_tfrecord_loc = False

            parser = sfio.tfrecords.get_tfrecord_parser(tfrecord,
                                                        ('image_raw', 'slide', 'loc_x', 'loc_y'),
                                                        normalizer=normalizer,
                                                        standardize=True,
                                                        img_size=self.image_size,
                                                        error_if_invalid=False) # Returns None for loc_x/loc_y if not in tfrecords

            dataset = dataset.map(parser, num_parallel_calls=8)
            dataset = dataset.batch(batch_size, drop_remainder=False)

            fl_activations_combined, logits_combined, slides_combined, loc_x_combined, loc_y_combined = [], [], [], [], []

            for i, data in enumerate(dataset):
                batch_processed_images, batch_slides, batch_loc_x, batch_loc_y = data
                batch_slides = batch_slides.numpy()
                batch_slides = np.array([unique_slides.index(bs.decode('utf-8')) for bs in batch_slides], dtype=np.uint32)

                model_output = combined_model.predict(batch_processed_images)

                if include_logits:
                    fl_activations, logits = model_output
                else:
                    fl_activations = model_output

                fl_activations_combined += [fl_activations.numpy()]
                slides_combined += [batch_slides]

                if include_logits:
                    logits_combined += [logits.numpy()]
                if include_tfrecord_loc:
                    loc_x_combined += [batch_loc_x.numpy()]
                    loc_y_combined += [batch_loc_y.numpy()]

                if log.getEffectiveLevel() <= 20:
                    name_str = f'\r(TFRecord {t+1:>3}/{len(self.tfrecords):>3})'
                    batch_str = f'(Batch {i+1:>3})'
                    img_str = f'({(i+1)*batch_size:>5} images)'
                    sys.stdout.write(f'{name_str} {batch_str} {img_str}: {sfutil.green(sfutil.path_to_name(tfrecord))}')
                    sys.stdout.flush()

                if max_tiles_per_slide and (i+1)*batch_size >= max_tiles_per_slide:
                    break

            fl_activations_combined = np.concatenate(fl_activations_combined)
            slides_combined = np.concatenate(slides_combined)

            if include_logits:
                logits_combined = np.concatenate(logits_combined)
            if include_tfrecord_loc:
                loc_x_combined = np.concatenate(loc_x_combined)
                loc_y_combined = np.concatenate(loc_y_combined)
                loc_combined = np.stack((loc_x_combined, loc_y_combined), axis=-1)
            else:
                loc_combined = None

            # Check if TFRecord was empty
            if fl_activations_combined == []:
                formatted_name = sfutil.green(sfutil.path_to_name(tfrecord))
                log.warning(f'Unable to calculate activations from {formatted_name}; is the TFRecord empty?')
                continue

            if not detected_logit_structure:
                nodes_names = [f'FLNode{f}' for f in range(fl_activations_combined.shape[1])]
                if include_logits:
                    logits_names = [f'Logits{l}' for l in range(logits_combined.shape[1])]
                else:
                    logits_names = []
                if export:
                    header = ['Slide'] + logits_names + nodes_names
                    csvwriter.writerow(header)
                for n in range(len(nodes_names)):
                    for slide in unique_slides:
                        self.slide_node_dict[slide].update({n: []})
                for l in range(len(logits_names)):
                    for slide in unique_slides:
                        self.slide_logits_dict[slide].update({l: []})
                detected_logit_structure=True

            if max_tiles_per_slide and ((i+1)*batch_size) > max_tiles_per_slide:
                slides_combined = slides_combined[:max_tiles_per_slide]
                fl_activations_combined = fl_activations_combined[:max_tiles_per_slide]
                logits_combined = logits_combined[:max_tiles_per_slide]
                if include_tfrecord_loc:
                    loc_combined = loc_combined[:max_tiles_per_slide]

            # Export to memory and CSV
            for i in range(len(fl_activations_combined)):
                slide = unique_slides[slides_combined[i]]

                activations_vals = fl_activations_combined[i]
                if include_logits:
                    logits_vals = logits_combined[i]

                # Write to CSV
                if export and include_logits:
                    row = [slide] + logits_vals.tolist() + activations_vals.tolist()
                    csvwriter.writerow(row)
                elif export:
                    row = [slide] + activations_vals.tolist()
                    csvwriter.writerow(row)

            # Write to memory
            for n in range(len(nodes_names)):
                node_activations = fl_activations_combined[:,n]
                if self.slide_node_dict[slide][n] == []:
                    self.slide_node_dict[slide][n] = node_activations
                else:
                    self.slide_node_dict[slide][n] = np.concatenate(self.slide_node_dict[slide][n], node_activations)
            if include_logits:
                for l in range(len(logits_names)):
                    logit_values = logits_combined[:, l]
                    if self.slide_logits_dict[slide][l] == []:
                        self.slide_logits_dict[slide][l] = logit_values
                    else:
                        self.slide_logits_dict[slide][l] = np.concatenate(self.slide_logits_dict[slide][l], logit_values)
            if loc_combined is not None and self.slide_loc_dict[slide] == []:
                self.slide_loc_dict[slide] = loc_combined
            elif loc_combined is not None:
                self.slide_loc_dict[slide] = np.concatenate(self.slide_loc_dict[slide], loc_combined)

        for slide in slides_to_remove:
            self.remove_slide(slide)

        if export:
            outfile.close()

        fla_calc_time = time.time()
        print('\r\033[K', end='')
        log.info(f'Activation calculation time: {fla_calc_time-fla_start_time:.0f} sec')
        log.info(f'Number of activation features: {self.num_features}')
        if export:
            log.info(f'Activations saved to {sfutil.green(export)}')

        # Dump PKL dictionary to file
        if cache:
            with open(cache, 'wb') as pt_pkl_file:
                pickle.dump([self.slide_node_dict, self.slide_logits_dict, self.slide_loc_dict], pt_pkl_file)
            log.info(f'Predictions and activations cached to {sfutil.green(cache)}')

        return self.slide_node_dict, self.slide_logits_dict

    def remove_slide(self, slide):
        del self.slide_node_dict[slide]
        del self.slide_logits_dict[slide]
        del self.slide_loc_dict[slide]
        self.tfrecords = [t for t in self.tfrecords if sfutil.path_to_name(t) != slide]
        try:
            self.slides.remove(slide)
        except ValueError:
            pass

    def export_to_csv(self, filename, method='mean', nodes=None):
        '''Exports calculated activations to csv.

        Args:
            filename:	Path to CSV file for export.
            nodes:		(optional) List of int. Activations of these nodes will be exported.
                            If None, activations for all nodes will be exported.
        '''
        with open(filename, 'w') as outfile:
            csvwriter = csv.writer(outfile)
            nodes = self.nodes if not nodes else nodes
            header = ['Slide'] + [f'FLNode{f}' for f in nodes]
            csvwriter.writerow(header)
            for slide in self.slide_node_dict:
                row = [slide]
                for n in nodes:
                    if method in ('mean', 'average', 'avg'):
                        row += [mean(self.slide_node_dict[slide][n])]
                    else:
                        row += [self.slide_node_dict[slide][n]]
                csvwriter.writerow(row)

    def export_to_torch(self, output_directory):
        import torch

        for slide in self.slide_node_dict:
            sys.stdout.write(f'\rWorking on {sfutil.green(slide)}...\033[K')
            sys.stdout.flush()
            slide_activations = []
            number_tiles = len(self.slide_node_dict[slide][self.nodes[0]])
            if number_tiles:
                for tile in range(number_tiles):
                    tile_activations = [self.slide_node_dict[slide][n][tile] for n in self.nodes]
                    slide_activations += [tile_activations]
                slide_activations = torch.from_numpy(np.array(slide_activations, dtype=np.float32))
                torch.save(slide_activations, join(output_directory, f'{slide}.pt'))
            else:
                print('\r\033[K', end='')
                log.info(f'Skipping empty slide {sfutil.green(slide)}')
        print('\r\033[K', end='')

        args = {
            'model': self.model,
            'num_features': self.num_features
        }
        sfutil.write_json(args, join(output_directory, 'settings.json'))

        log.info('Activations exported in Torch format.')

    def calculate_activation_averages_and_stats(self, filename=None, node_method='avg', threshold=0.5):
        '''Calculates activation averages across categories,
            as well as tile-level and patient-level statistics using ANOVA,
            exporting to CSV if desired.

        Args:
            filename:		Path to CSV file for export.
            node_method:	Either 'avg' (default) or 'threshold'. If avg, slide-level node data is calculated
                                by averaging node activations across all tiles. If threshold, slide-level node data
                                is calculated by counting the number of tiles with node activations > threshold
                                and dividing by the total number of tiles.

        Returns:
            Dict mapping slides to dict of nodes mapping to slide-level node values;
            Dict mapping nodes to tile-level dict of statistics ('p', 'f');
            Dict mapping nodes to slide-level dict of statistics ('p', 'f');
        '''
        if not self.categories:
            log.warning('Unable to calculate activations statistics; Please load annotations with load_annotations().')
            return
        if node_method not in ('avg', 'threshold'):
            raise ActivationsError(f"'node_method' must be either 'avg' or 'threshold', not {node_method}")

        empty_category_dict = {}
        slide_node_dict = {}
        tile_node_stats = {}
        slide_node_stats = {}
        for category in self.categories:
            empty_category_dict.update({category: []})
        if not hasattr(self, 'nodes'):
            log.error('Activations have not been generated, unable to calculate averages')
            return
        for node in self.nodes:
            self.node_cat_dict.update({node: deepcopy(empty_category_dict)})

        self.categories = self.categories[::-1]

        log.info('Calculating activation averages & stats across nodes...')
        for node in self.nodes:
            # For each node, calculate average across tiles found in a slide
            for slide in self.slides:
                if slide not in slide_node_dict: slide_node_dict.update({slide: {}})
                pt_cat = self.slide_category_dict[slide]
                if node_method == 'avg':
                    node_val = mean(self.slide_node_dict[slide][node])
                elif node_method == 'threshold':
                    sum_of_nodes = sum([1 for t in self.slide_node_dict[slide][node] if t > threshold])
                    len_of_nodes = len(self.slide_node_dict[slide][node])
                    node_val = sum_of_nodes / len_of_nodes
                self.node_cat_dict[node][pt_cat] += [node_val]
                slide_node_dict[slide][node] = node_val

            # Tile-level ANOVA
            fvalue, pvalue = stats.f_oneway(*self.get_tile_node_activations_by_category(node))
            if not isnan(fvalue) and not isnan(pvalue):
                tile_node_stats.update({node: {'f': fvalue,
                                          'p': pvalue} })
            else:
                tile_node_stats.update({node: {'f': -1,
                                          'p': 1} })

            # Patient-level ANOVA
            fvalue, pvalue = stats.f_oneway(*[self.node_cat_dict[node][c] for c in self.used_categories])
            if not isnan(fvalue) and not isnan(pvalue):
                slide_node_stats.update({node: {'f': fvalue,
                                                 'p': pvalue} })
            else:
                slide_node_stats.update({node: {'f': -1,
                                                   'p': 1} })

        try:
            self.nodes = sorted(self.nodes, key=lambda n: tile_node_stats[n]['p'])
            self.sorted_nodes = sorted(self.nodes, key=lambda n: slide_node_stats[n]['p'])
        except:
            log.warning('No stats calculated; unable to sort nodes.')
            self.sorted_nodes = self.nodes

        for i, node in enumerate(self.nodes):
            if self.focus_nodes and (node not in self.focus_nodes): continue
            try:
                log.info(f"Tile-level P-value ({node}): {tile_node_stats[node]['p']}")
                log.info(f"Patient-level P-value: ({node}): {slide_node_stats[node]['p']}")
            except:
                log.warning(f'No stats calculated for node {node}')
            if (not self.focus_nodes) and i>9: break

        # Export results
        export_file = join(self.export_dir, 'slide_level_summary.csv') if not filename else filename
        self._save_node_statistics_to_csv(self.sorted_nodes,
                                          slide_node_dict,
                                          filename=export_file,
                                          tile_stats=tile_node_stats,
                                          slide_stats=slide_node_stats)
        return slide_node_dict, tile_node_stats, slide_node_stats

    def logistic_regression(self, slide_method='avg', node_threshold=0.5):
        '''Experimental function, creates a logistic regression model
            to generate slide-level predictions from slide-level statistics.'''
        slide_node_dict, _, _ = self.calculate_activation_averages_and_stats(None, slide_method, node_threshold)

        log.info('Working on logistic regression...')
        x = np.array([[slide_node_dict[slide][n] for n in self.nodes] for slide in self.slides])
        y = np.array([self.categories.index(self.slide_category_dict[slide]) for slide in self.slides])

        model = LogisticRegression(solver='lbfgs', max_iter=100, multi_class='ovr').fit(x, y)
        log.info(f'Regression complete, accuracy: {model.score(x, y):.3f}')
        return model

    def generate_box_plots(self, export_folder=None):
        '''Generates box plots comparing nodal activations at the slide-level and tile-level.

        Args:
            export_folder:	(optional) Path to directory in which to save box plots.
                                If None, will save boxplots to STATS_ROOT directory.
        '''

        if not self.categories:
            log.warning('Unable to generate box plots; annotations not loaded. Please load with load_annotations().')
            return

        # First ensure basic stats have been calculated
        if not self.sorted_nodes:
            self.calculate_activation_averages_and_stats()
        if not export_folder: export_folder = self.export_dir

        # Display tile-level box plots & stats
        log.info('Generating box plots...')
        for i, node in enumerate(self.nodes):
            if self.focus_nodes and (node not in self.focus_nodes): continue
            plt.clf()
            snsbox = sns.boxplot(data=self.get_tile_node_activations_by_category(node))
            title = f'{node} (tile-level)'
            snsbox.set_title(title)
            snsbox.set(xlabel='Category', ylabel='Activation')
            plt.xticks(plt.xticks()[0], self.used_categories)
            boxplot_filename = join(export_folder, f'boxplot_{title}.png')
            plt.gcf().canvas.start_event_loop(sys.float_info.min)
            plt.savefig(boxplot_filename, bbox_inches='tight')
            if (not self.focus_nodes) and i>4: break

        # Print slide_level box plots & stats
        for i, node in enumerate(self.sorted_nodes):
            if self.focus_nodes and (node not in self.focus_nodes): continue
            plt.clf()
            snsbox = sns.boxplot(data=[self.node_cat_dict[node][c] for c in self.used_categories])
            title = f'{node} (slide-level)'
            snsbox.set_title(title)
            snsbox.set(xlabel='Category',ylabel='Average tile activation')
            plt.xticks(plt.xticks()[0], self.used_categories)
            boxplot_filename = join(export_folder, f'boxplot_{title}.png')
            plt.gcf().canvas.start_event_loop(sys.float_info.min)
            plt.savefig(boxplot_filename, bbox_inches='tight')
            if (not self.focus_nodes) and i>4: break

    def save_example_tiles_gradient(self, nodes=None, export_folder=None, tile_filter=None):
        '''For a given set of activation nodes, saves image tiles named according
            to their corresponding node activations, for easy sorting and visualization.
            Duplicate image tiles will be saved for each node, organized into subfolders named according to node id.

        Args:
            nodes:			List of int, nodes to evaluate
            export_folder:	Path to folder in which to save examples tiles
            tile_filter:	(optional) Dict mapping slide names to tile indices.
                                If provided, will only save image tiles from this list.
                                Example:
                                {'slide1': [0, 16, 200]}
        '''
        if not export_folder: export_folder = join(self.export_dir, 'sorted_tiles')
        if not nodes: nodes = self.focus_nodes

        for node in nodes:
            if not exists(join(export_folder, node)):
                os.makedirs(join(export_folder, node))

            gradient = []
            for slide in self.slides:
                for i, tile in enumerate(self.slide_node_dict[slide][node]):
                    if tile_filter and (slide not in tile_filter) or (i not in tile_filter[slide]):
                        continue
                    gradient += [{
                                    'val': tile,
                                    'slide': slide,
                                    'index': i
                    }]
            gradient = sorted(gradient, key=lambda k: k['val'])
            for i, g in enumerate(gradient):
                print(f'Extracting tile {i} of {len(gradient)} for node {node}...', end='\r')
                for tfr in self.tfrecords:
                    if sfutil.path_to_name(tfr) == g['slide']:
                        tfr_dir = tfr
                if not tfr_dir:
                    log.warning(f"TFRecord location not found for slide {g['slide']}")
                slide, image = sfio.tfrecords.get_tfrecord_by_index(tfr_dir, g['index'], decode=False)
                tile_filename = f"{i}-tfrecord{g['slide']}-{g['index']}-{g['val']:.2f}.jpg"
                image_string = open(join(export_folder, node, tile_filename), 'wb')
                image_string.write(image.numpy())
                image_string.close()
            print()

    def save_example_tiles_high_low(self, nodes, slides, export_folder=None):
        '''For a given set of activation nodes, saves images of tiles with the highest and lowest
        activations in these nodes, restricted to the set of slides designated.

        Args:
            nodes:			List of int. Nodes with which to perform this function.
            slides:			List of slide names. Will load tile images from these slides.
            export_folder:	Path to directory in which to save image tiles.
        '''
        if not export_folder: export_folder = join(self.export_dir, 'example_tiles')
        for node in nodes:
            for slide in slides:
                sorted_index = np.argsort(self.slide_node_dict[slide][node])
                lowest = sorted_index[:10]
                highest = sorted_index[-10:]
                lowest_dir = join(export_folder, node, slide, 'lowest')
                highest_dir = join(export_folder, node, slide, 'highest')
                if not exists(lowest_dir): os.makedirs(lowest_dir)
                if not exists(highest_dir): os.makedirs(highest_dir)

                for tfr in self.tfrecords:
                    if sfutil.path_to_name(tfr) == slide:
                        tfr_dir = tfr
                if not tfr_dir:
                    log.warning(f'TFRecord location not found for slide {slide}')

                def extract_by_index(indices, directory):
                    for index in indices:
                        slide, image = sfio.tfrecords.get_tfrecord_by_index(tfr_dir, index, decode=False)
                        tile_filename = f'tfrecord{slide.numpy()}-tile{index}.jpg'
                        image_string = open(join(directory, tile_filename), 'wb')
                        image_string.write(image.numpy())
                        image_string.close()

                extract_by_index(lowest, lowest_dir)
                extract_by_index(highest, highest_dir)

class TileVisualizer:
    '''Class to supervize visualization of node activations across an image tile.
    Visualization is accomplished by performing sequential convolutional masking
        and determining impact of masking on node activation. In this way,
        the masking reveals spatial importance with respect to activation of the given node.
    '''

    def __init__(self,
                 model,
                 node,
                 tile_px,
                 mask_width=None,
                 normalizer=None,
                 normalizer_source=None):

        '''Object initializer.

        Args:
            model:				Path to Tensorflow model
            node:				Int, activation node to analyze
            tile_px:			Int, width/height of image tiles
            mask_width:			Width of mask to convolutionally apply. Defaults to 1/6 of tile_px
            normalizer:			String, normalizer to apply to tiles in real-time.
            normalizer_source:	Path to normalizer source image.
        '''
        self.NODE = node
        self.IMAGE_SHAPE = (tile_px, tile_px, 3)
        self.MASK_WIDTH = mask_width if mask_width else int(self.IMAGE_SHAPE[0]/6)
        self.normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

        log.info('Initializing tile visualizer')
        log.info(f'Node: {sfutil.bold(str(node))} | Shape: ({self.IMAGE_SHAPE}) | Window size: {self.MASK_WIDTH}')
        log.info(f'Loading Tensorflow model at {sfutil.green(model)}...')

        self.loaded_model = ModelActivationsInterface(model)

    def _calculate_activation_map(self, stride_div=4):
        '''Creates map of importance through convolutional masking and
        examining changes in node activations.'''
        sx = self.IMAGE_SHAPE[0]
        sy = self.IMAGE_SHAPE[1]
        w  = self.MASK_WIDTH
        stride = int(self.MASK_WIDTH / stride_div)
        min_x  = int(w/2)
        max_x  = int(sx - w/2)
        min_y  = int(w/2)
        max_y  = int(sy - w/2)

        act_array = []
        for yi in range(min_y, max_y, stride):
            for xi in range(min_x, max_x, stride):
                mask = self._create_bool_mask(xi, yi, w, sx, sy)
                masked = self.tf_processed_image.numpy() * mask
                act, _ = self.loaded_model.predict(np.array([masked]))
                act_array += [act[0][self.NODE]]
                print(f'Calculating activations at x:{xi}, y:{yi}; act={act[0][self.NODE]}', end='\033[K\r')
        max_center_x = max(range(min_x, max_x, stride))
        max_center_y = max(range(min_y, max_y, stride))
        reshaped_array = np.reshape(np.array(act_array), [len(range(min_x, max_x, stride)),
                                                          len(range(min_y, max_y, stride))])
        print()
        return reshaped_array, max_center_x, max_center_y

    def _create_bool_mask(self, x, y, w, sx, sy):
        l = max(0,  int(x-(w/2.)))
        r = min(sx, int(x+(w/2.)))
        t = max(0,  int(y-(w/2.)))
        b = min(sy, int(y+(w/2.)))
        m = np.array([[[True]*3]*sx]*sy)
        for yi in range(m.shape[1]):
            for xi in range(m.shape[0]):
                if (t < yi < b) and (l < xi < r):
                    m[yi][xi] = [False, False, False]
        return m

    def _predict_masked(self, x, y, index):
        mask = self._create_bool_mask(x, y, self.MASK_WIDTH, self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1])
        masked = self.tf_processed_image.numpy() * mask
        act, _ = self.loaded_model.predict(np.array([masked]))
        return act[0][index]

    def visualize_tile(self,
                       tfrecord=None,
                       index=None,
                       image_jpg=None,
                       export_folder=None,
                       zoomed=True,
                       interactive=False):

        '''Visualizes tiles, either interactively or saving to directory.

        Args:
            tfrecord:			If provided, will visualize tile from the designated tfrecord.
                                    Must supply either a tfrecord and index or image_jpg
            index:				Index of tile to visualize within tfrecord, if provided
            image_jpeg:			JPG image to perform analysis on
            export_folder:		Folder in which to save heatmap visualization
            zoomed:				Bool. If true, will crop image to space containing heatmap
                                    (otherwise a small border will be seen)
            interactive:		If true, will display as interactive map using matplotlib
        '''
        if not (image_jpg or tfrecord):
            raise ActivationsError('Must supply either tfrecord or image_jpg')

        if image_jpg:
            log.info(f'Processing tile at {sfutil.green(image_jpg)}...')
            tilename = sfutil.path_to_name(image_jpg)
            self.tile_image = Image.open(image_jpg)
            image_file = open(image_jpg, 'rb')
            tf_decoded_image = tf.image.decode_png(image_file.read(), channels=3)
        else:
            slide, tf_decoded_image = sfio.tfrecords.get_tfrecord_by_index(tfrecord, index, decode=True)
            tilename = f"{slide.numpy().decode('utf-8')}-{index}"
            self.tile_image = Image.fromarray(tf_decoded_image.numpy())

        # Normalize PIL image & TF image
        if self.normalizer:
            self.tile_image = self.normalizer.pil_to_pil(self.tile_image)
            tf_decoded_image = tf.py_function(self.normalizer.tf_to_rgb, [self.tile_image], tf.int32)

        # Next, process image with Tensorflow
        self.tf_processed_image = tf.image.per_image_standardization(tf_decoded_image)
        self.tf_processed_image = tf.image.convert_image_dtype(self.tf_processed_image, tf.float16)
        self.tf_processed_image.set_shape(self.IMAGE_SHAPE)

        # Now create the figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.implot = plt.imshow(self.tile_image)

        if interactive:
            self.rect = patches.Rectangle((0, 0), self.MASK_WIDTH, self.MASK_WIDTH, facecolor='white', zorder=20)
            self.ax.add_patch(self.rect)

        activation_map, max_center_x, max_center_y = self._calculate_activation_map()

        # Prepare figure
        filename = join(export_folder, f'{tilename}-heatmap.png')

        def hover(event):
            if event.xdata:
                self.rect.set_xy((event.xdata-self.MASK_WIDTH/2, event.ydata-self.MASK_WIDTH/2))
                print(self._predict_masked(event.xdata, event.ydata, index=self.NODE), end='\r')
                self.fig.canvas.draw_idle()

        def click(event):
            if event.button == 1:
                self.MASK_WIDTH = min(min(self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1]), self.MASK_WIDTH + 25)
                self.rect.set_width(self.MASK_WIDTH)
                self.rect.set_height(self.MASK_WIDTH)
            else:
                self.MASK_WIDTH = max(0, self.MASK_WIDTH - 25)
                self.rect.set_width(self.MASK_WIDTH)
                self.rect.set_height(self.MASK_WIDTH)
            self.fig.canvas.draw_idle()

        if interactive:
            self.fig.canvas.mpl_connect('motion_notify_event', hover)
            self.fig.canvas.mpl_connect('button_press_event', click)

        if activation_map is not None:
            # Calculate boundaries of heatmap
            hw = int(self.MASK_WIDTH/2)
            if zoomed:
                extent = (hw, max_center_x, max_center_y, hw)
            else:
                extent = (0, max_center_x+hw, max_center_y+hw, 0)

            # Heatmap
            divnorm = mcol.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)
            self.ax.imshow(activation_map,
                           extent=extent,
                           cmap='coolwarm',
                           norm=divnorm,
                           alpha=0.6 if not interactive else 0.0,
                           interpolation='bicubic',
                           zorder=10)
        if filename:
            plt.savefig(filename, bbox_inches='tight')
            log.info(f'Heatmap saved to {filename}')
        if interactive:
            plt.show()

class Heatmap:
    '''Generates heatmap by calculating predictions from a sliding scale window across a slide.'''

    def __init__(self,
                 slide_path,
                 model_path,
                 tile_px,
                 tile_um,
                 stride_div=2,
                 roi_dir=None,
                 roi_list=None,
                 roi_method='inside',
                 buffer=True,
                 normalizer=None,
                 normalizer_source=None,
                 batch_size=16,
                 num_threads=8):

        '''Convolutes across a whole slide, calculating logits and saving predictions internally for later use.

        Args:
            slide_path:			Path to slide
            model_path:			Path to Tensorflow model
            tile_px:			Size of image tiles, in pixels
            tile_um:			Size of image tiles, in microns
            stride_div:			Divisor for stride when convoluting across slide
            roi_dir:			Directory in which slide ROI is contained
            roi_list:			If a roi_dir is not supplied, a list of paths to ROI CSVs can be provided
            roi_method:			Either 'inside', 'outside', or 'ignore'.
                                    If inside, tiles will be extracted inside ROI region
                                    If outside, tiles will be extracted outside ROI region
            buffer:				Either 'vmtouch' or path to directory to use for buffering slides
                                    Significantly improves performance for slides on HDDs
            normalizer:			Normalization strategy to use on image tiles
            normalizer_source:	Path to normalizer source image
            batch_size:			Batch size when calculating predictions
        '''
        from slideflow.slide import WSI

        self.logits = None
        self.tile_px = tile_px
        self.tile_um = tile_um

        # Setup normalization
        self.normalizer = normalizer
        self.normalizer_source = normalizer_source

        # Load the designated model
        self.model = ModelActivationsInterface(model_path)

        # Create slide buffer
        if buffer and os.path.isdir(buffer):
            buffered_slide = True
            new_path = os.path.join(buffer, os.path.basename(slide_path))
            shutil.copy(slide_path, new_path)
            slide_path = new_path
        else:
            buffered_slide = False

        # Load the slide
        self.slide = WSI(slide_path,
                         tile_px,
                         tile_um,
                         stride_div,
                         enable_downsample=False,
                         roi_dir=roi_dir,
                         roi_list=roi_list,
                         roi_method=roi_method,
                         silent=True,
                         buffer=buffer,
                         skip_missing_roi=(roi_method == 'inside'))

        # Record the number of classes in the model
        self.num_classes = self.model.num_classes #_model.layers[-1].output_shape[-1]

        if not self.slide.loaded_correctly():
            raise ActivationsError(f'Unable to load slide {self.slide.name} for heatmap generation')

        # Pre-load thumbnail in separate thread
        thumb_process = DProcess(target=partial(self.slide.thumb, width=2048))
        thumb_process.start()

        # Create tile coordinate generator
        gen_slice = self.slide.build_generator(normalizer=self.normalizer,
                                               normalizer_source=self.normalizer_source,
                                               include_loc=False,
                                               shuffle=False,
                                               num_threads=num_threads,
                                               show_progress=True)

        if not gen_slice:
            log.error(f'No tiles extracted from slide {sfutil.green(self.slide.name)}')

        # Generate dataset from the generator
        with tf.name_scope('dataset_input'):
            output_signature = {'image':tf.TensorSpec(shape=(tile_px,tile_px,3), dtype=tf.int32)}
            tile_dataset = tf.data.Dataset.from_generator(gen_slice, output_signature=output_signature)
            tile_dataset = tile_dataset.map(self._parse_function, num_parallel_calls=8)
            tile_dataset = tile_dataset.batch(batch_size, drop_remainder=False)
            tile_dataset = tile_dataset.prefetch(8)

        # Iterate through generator to calculate logits +/- final layer activations for all tiles
        logits_arr = []		# Logits (predictions)
        postconv_arr = []	# Post-convolutional layer (post-convolutional activations)
        for batch_images in tile_dataset:
            postconv, logits = self.model.predict(batch_images)
            logits_arr += [logits]
            postconv_arr += [postconv]
        logits_arr = np.concatenate(logits_arr)
        postconv_arr = np.concatenate(postconv_arr)
        num_postconv_nodes = postconv_arr.shape[1]

        log.info('Finished predictions. Waiting on thumbnail...')
        thumb_process.join()

        if ((self.slide.tile_mask is not None) and
             (self.slide.extracted_x_size) and
             (self.slide.extracted_y_size) and
             (self.slide.full_stride)):

            # Expand logits back to a full 2D map spanning the whole slide,
            #  supplying values of '0' where tiles were skipped by the tile generator
            x_logits_len = int(self.slide.extracted_x_size / self.slide.full_stride) + 1
            y_logits_len = int(self.slide.extracted_y_size / self.slide.full_stride) + 1
            expanded_logits = [[-1] * self.num_classes] * len(self.slide.tile_mask)
            expanded_postconv = [[-1] * num_postconv_nodes] * len(self.slide.tile_mask)
            li = 0
            for i in range(len(expanded_logits)):
                if self.slide.tile_mask[i] == 1:
                    expanded_logits[i] = logits_arr[li]
                    expanded_postconv[i] = postconv_arr[li]
                    li += 1

            expanded_logits = np.asarray(expanded_logits, dtype=float)
            expanded_postconv = np.asarray(expanded_postconv, dtype=float)

            # Resize logits array into a two-dimensional array for heatmap display
            self.logits = np.resize(expanded_logits, [y_logits_len, x_logits_len, self.num_classes])
            self.postconv = np.resize(expanded_postconv, [y_logits_len, x_logits_len, num_postconv_nodes])
        else:
            self.logits = logits_arr
            self.postconv = postconv_arr

        if (type(self.logits) == bool) and (not self.logits):
            log.error(f'Unable to create heatmap for slide {sfutil.green(self.slide.name)}')

        # Unbuffer slide
        if buffered_slide:
            os.remove(new_path)

    def _parse_function(self, record):
        image = record['image']
        parsed_image = tf.image.per_image_standardization(image)
        parsed_image = tf.image.convert_image_dtype(parsed_image, tf.float32)
        parsed_image.set_shape([self.tile_px, self.tile_px, 3])
        return parsed_image

    def _prepare_figure(self, show_roi=True):
        self.fig = plt.figure(figsize=(18, 16))
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom = 0.25, top=0.95)
        gca = plt.gca()
        gca.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)
        # Plot ROIs
        if show_roi:
            print('\r\033[KPlotting ROIs...', end='')
            ROI_SCALE = self.slide.full_shape[0]/2048
            annPolys = [sg.Polygon(annotation.scaled_area(ROI_SCALE)) for annotation in self.slide.rois]
            for poly in annPolys:
                x,y = poly.exterior.xy
                plt.plot(x, y, zorder=20, color='k', linewidth=5)

    def display(self, show_roi=True, interpolation='none', logit_cmap=None):
        '''Interactively displays calculated logits as a heatmap.

        Args:
            show_roi:			Bool, whether to overlay ROIs onto heatmap image
            interpolation:		Interpolation strategy to use for smoothing heatmap
            logit_cmap:			Either function or a dictionary use to create heatmap colormap.
                                    Each image tile will generate a list of predictions of length O,
                                    where O is the number of outcome categories.
                                    If logit_cmap is a function, then the logit prediction list will be passed
                                        to the function, and the function is expected to return [R, G, B] values
                                        which will be displayed.
                                    If the logit_cmap is a dictionary, it should map 'r', 'g', and 'b' to indices;
                                        The prediction for these outcome indices will be mapped to the RGB colors.
                                        Thus, the corresponding color will only reflect up to three outcomes.
                                        Example mapping prediction for outcome 0 to the red colorspace, 3 to green, etc
                                        {
                                            'r': 0,
                                            'g': 3,
                                            'b': 1
                                        }
        '''
        self._prepare_figure(show_roi=False)
        heatmap_dict = {}

        if show_roi: thumb = self.slide.annotated_thumb()
        else: thumb = self.slide.thumb()
        implot = FastImshow(thumb, self.ax, extent=None, tgt_res=1024)

        def slider_func(val):
            for h, s in heatmap_dict.values():
                h.set_alpha(s.val)

        if logit_cmap:
            if callable(logit_cmap):
                map_logit = logit_cmap
            else:
                def map_logit(l):
                    # Make heatmap with specific logit predictions mapped to r, g, and b
                    return (l[logit_cmap['r']], l[logit_cmap['g']], l[logit_cmap['b']])
            heatmap = self.ax.imshow([[map_logit(l) for l in row] for row in self.logits],
                                     extent=implot.extent,
                                     interpolation=interpolation,
                                     zorder=10)
        else:
            divnorm = mcol.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)
            for i in range(self.num_classes):
                heatmap = self.ax.imshow(self.logits[:, :, i],
                                         extent=implot.extent,
                                         cmap='coolwarm',
                                         norm=divnorm,
                                         alpha = 0.0,
                                         interpolation=interpolation,
                                         zorder=10) #bicubic

                ax_slider = self.fig.add_axes([0.25, 0.2-(0.2/self.num_classes)*i, 0.5, 0.03], facecolor='lightgoldenrodyellow')
                slider = Slider(ax_slider, f'Class {i}', 0, 1, valinit = 0)
                heatmap_dict.update({f'Class{i}': [heatmap, slider]})
                slider.on_changed(slider_func)

        self.fig.canvas.set_window_title(self.slide.name)
        implot.show()
        plt.show()

    def save(self, save_folder, show_roi=True, interpolation='none', logit_cmap=None, vmin=0, vmax=1, vcenter=0.5):
        '''Saves calculated logits as heatmap overlays.

        Args:
            show_roi:			Bool, whether to overlay ROIs onto heatmap image
            interpolation:		Interpolation strategy to use for smoothing heatmap
            logit_cmap:			Either function or a dictionary use to create heatmap colormap.
                                    Each image tile will generate a list of predictions of length O,
                                    where O is the number of outcome categories.
                                    If logit_cmap is a function, then the logit prediction list will be passed
                                        to the function, and the function is expected to return [R, G, B] values
                                        which will be displayed.
                                    If the logit_cmap is a dictionary, it should map 'r', 'g', and 'b' to indices;
                                        The prediction for these outcome indices will be mapped to the RGB colors.
                                        Thus, the corresponding color will only reflect up to three outcomes.
                                        Example mapping prediction for outcome 0 to the red colorspace, 3 to green, etc
                                        {
                                            'r': 0,
                                            'g': 3,
                                            'b': 1
                                        }
        '''
        print('\r\033[KSaving base figures...', end='')

        # Save base thumbnail as separate figure
        self._prepare_figure(show_roi=False)
        self.ax.imshow(self.slide.thumb(width=2048), zorder=0)
        plt.savefig(os.path.join(save_folder, f'{self.slide.name}-raw.png'), bbox_inches='tight')
        plt.clf()

        # Save thumbnail + ROI as separate figure
        self._prepare_figure(show_roi=False)
        self.ax.imshow(self.slide.annotated_thumb(width=2048), zorder=0)
        plt.savefig(os.path.join(save_folder, f'{self.slide.name}-raw+roi.png'), bbox_inches='tight')
        plt.clf()

        # Now prepare base image for the the heatmap overlay
        self._prepare_figure(show_roi=False)
        thumb_func = self.slide.annotated_thumb if show_roi else self.slide.thumb
        implot = self.ax.imshow(thumb_func(width=2048), zorder=0)

        if logit_cmap:
            if callable(logit_cmap):
                map_logit = logit_cmap
            else:
                def map_logit(l):
                    # Make heatmap with specific logit predictions mapped to r, g, and b
                    return (l[logit_cmap['r']], l[logit_cmap['g']], l[logit_cmap['b']])

            heatmap = self.ax.imshow([[map_logit(l) for l in row] for row in self.logits],
                                     extent=implot.get_extent(),
                                     interpolation=interpolation,
                                     zorder=10)

            plt.savefig(os.path.join(save_folder, f'{self.slide.name}-custom.png'), bbox_inches='tight')
        else:
            # Make heatmap plots and sliders for each outcome category
            for i in range(self.num_classes):
                print(f'\r\033[KMaking heatmap {i+1} of {self.num_classes}...', end='')
                divnorm = mcol.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                heatmap = self.ax.imshow(self.logits[:, :, i],
                                         extent=implot.get_extent(),
                                         cmap='coolwarm',
                                         norm=divnorm,
                                         vmin=vmin,
                                         vmax=vmax,
                                         alpha=0.6,
                                         interpolation=interpolation, #bicubic
                                         zorder=10)
                plt.savefig(os.path.join(save_folder, f'{self.slide.name}-{i}.png'), bbox_inches='tight')
                heatmap.set_alpha(1)
                plt.savefig(os.path.join(save_folder, f'{self.slide.name}-{i}-solid.png'), bbox_inches='tight')
                heatmap.remove()

        plt.close()
        print('\r\033[K', end='')
        log.info(f'Saved heatmaps for {sfutil.green(self.slide.name)}')
