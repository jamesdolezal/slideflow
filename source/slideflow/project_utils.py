'''Utility functions for SlideflowProject, primarily for use in the context of multiprocessing.'''

import os
import sys
import types
from os.path import join, exists, basename
import slideflow.util as sfutil
import slideflow.io as sfio

from slideflow.io import Dataset
from slideflow.util import log

def get_validation_settings(**kwargs):
    '''
    Returns a namespace of validation settings.

    Args:
        target: 			(default: 'per-patient') Whether to select validation data
                                on a 'per-patient' (default) or 'per-tile' basis.
        strategy: 			(default: 'k-fold') Validation dataset selection strategy
                                (bootstrap, k-fold, k-fold-manual, k-fold-preserved-site, fixed, none).
        k_fold: 			(default: 3) Number of k-folds, if strategy is 'k-fold'
        k_fold_iter: 		K iteration, if using k-folds. Starts at 1.
        k_fold_header: 		Annotations file header column for manually specifying k-fold.
                                Only used if validation_strategy is 'k-fold-manual'
        fraction: 			Fraction of data to use for validation testing, if strategy is "fixed"
        dataset: 			If specified, will use this dataset for validation
        annotations: 		If specified, will use this annotations file for validation
        filters:			If specified, will use these filters for validation
        validate_on_batch:	Validation will be performed every N batches.
        validation_steps:	Number of steps of validation to perform each time doing a validation check.
    '''

    args_dict = {
        'target': 'per-patient',
        'strategy': 'k-fold',
        'k_fold': 3,
        'k_fold_iter': None,
        'k_fold_header': None,
        'fraction': None,
        'dataset': None,
        'annotations': None,
        'filters': None,
        'validate_on_batch': 512,
        'validation_steps': 200,
        'batch_size': 64
    }
    for k in kwargs:
        args_dict[k] = kwargs[k]
    args = types.SimpleNamespace(**args_dict)

    if (args.k_fold_header is None and args.strategy == 'k-fold-manual'):
        raise Exception("Must supply 'k_fold_header' if validation strategy is 'k-fold-manual'")

    return args

def print_fn(string):
    sys.stdout.write(f'\r\033[K{string}\n')
    sys.stdout.flush()

def tile_extractor(slide_path, roi_dir, roi_method, skip_missing_roi, randomize_origin,
                    img_format, tma, full_core, shuffle, tile_px, tile_um, stride_div,
                    downsample, whitespace_fraction, whitespace_threshold, grayspace_fraction,
                    grayspace_threshold, normalizer, normalizer_source, split_fraction,
                    split_names, report_dir, tfrecord_dir, tiles_dir, save_tiles, save_tfrecord,
                    buffer, threads_per_worker, pb_counter, counter_lock):

    from slideflow.slide import TMAReader, SlideReader, TileCorruptionError
    try:
        log.empty(f'Exporting tiles for slide {sfutil.path_to_name(slide_path)}', 1, print_fn)

        if tma:
            whole_slide = TMAReader(slide_path,
                                    tile_px,
                                    tile_um,
                                    stride_div,
                                    enable_downsample=downsample,
                                    report_dir=report_dir,
                                    buffer=buffer)
        else:
            whole_slide = SlideReader(slide_path,
                                    tile_px,
                                    tile_um,
                                    stride_div,
                                    enable_downsample=downsample,
                                    roi_dir=roi_dir,
                                    roi_method=roi_method,
                                    randomize_origin=randomize_origin,
                                    skip_missing_roi=skip_missing_roi,
                                    buffer=buffer,
                                    pb_counter=pb_counter,
                                    counter_lock=counter_lock,
                                    print_fn=print_fn)

        if not whole_slide.loaded_correctly():
            return

        try:
            report = whole_slide.extract_tiles(tfrecord_dir=tfrecord_dir if save_tfrecord else None,
                                                tiles_dir=tiles_dir if save_tiles else None,
                                                split_fraction=split_fraction,
                                                split_names=split_names,
                                                whitespace_fraction=whitespace_fraction,
                                                whitespace_threshold=whitespace_threshold,
                                                grayspace_fraction=grayspace_fraction,
                                                grayspace_threshold=grayspace_threshold,
                                                normalizer=normalizer,
                                                normalizer_source=normalizer_source,
                                                img_format=img_format,
                                                full_core=full_core,
                                                shuffle=shuffle,
                                                num_threads=threads_per_worker)
        except TileCorruptionError:
            if downsample:
                formatted_path = sfutil.green(sfutil.path_to_name(slide_path))
                log.warn(f'Corrupt tile in {formatted_path}; will try disabling downsampling', 1, print_fn)
                report = tile_extractor(
                    slide_path,
                    roi_dir,
                    roi_method,
                    skip_missing_roi,
                    randomize_origin,
                    img_format,
                    tma,
                    full_core,
                    shuffle,
                    tile_px,
                    tile_um,
                    stride_div,
                    False, #downsample = False
                    whitespace_fraction,
                    whitespace_threshold,
                    grayspace_fraction,
                    grayspace_threshold,
                    normalizer,
                    normalizer_source,
                    split_fraction,
                    split_names,
                    report_dir,
                    tfrecord_dir,
                    tiles_dir,
                    save_tiles,
                    save_tfrecord,
                    buffer,
                    threads_per_worker,
                    pb_counter,
                    counter_lock)
            else:
                log.error(f'Corrupt tile in {sfutil.green(sfutil.path_to_name(slide_path))}; skipping slide', 1, print_fn)
                return
        del whole_slide
        return report
    except (KeyboardInterrupt, SystemExit):
        print('Exiting...')
        return

def activations_generator(project, model, outcome_label_headers=None, layers=None, filters=None, filter_blank=None,
                          focus_nodes=None, activations_export=None, activations_cache='default', normalizer=None,
                          normalizer_source=None, max_tiles_per_slide=100, min_tiles_per_slide=None,
                          include_logits=True, batch_size=None, torch_export=None, results_dict=None):

    from slideflow.activations import ActivationsVisualizer

    log.header('Generating layer activations...')
    layers = [layers] if not isinstance(layers, list) else layers

    # Setup directories
    stats_root = join(project.root, 'stats')
    if not exists(stats_root): os.makedirs(stats_root)

    # Load dataset for evaluation
    hp_data = sfutil.get_model_hyperparameters(model)
    activations_dataset = Dataset(config_file=project.dataset_config,
                                    sources=project.datasets,
                                    tile_px=hp_data['hp']['tile_px'],
                                    tile_um=hp_data['hp']['tile_um'],
                                    annotations=project.annotations,
                                    filters=filters,
                                    filter_blank=filter_blank)

    tfrecords_list = activations_dataset.get_tfrecords(ask_to_merge_subdirs=True)

    log.info(f'Visualizing activations from {len(tfrecords_list)} slides', 1)

    if activations_export:
        activations_export = join(stats_root, activations_export)

    AV = ActivationsVisualizer(model=model,
                                tfrecords=tfrecords_list,
                                export_dir=join(project.root, 'stats'),
                                image_size=hp_data['hp']['tile_px'],
                                annotations=project.annotations,
                                outcome_label_headers=outcome_label_headers,
                                focus_nodes=focus_nodes,
                                normalizer=normalizer,
                                normalizer_source=normalizer_source,
                                activations_export=activations_export,
                                cache=activations_cache,
                                max_tiles_per_slide=max_tiles_per_slide,
                                min_tiles_per_slide=min_tiles_per_slide,
                                manifest=activations_dataset.get_manifest(),
                                layers=layers,
                                include_logits=include_logits,
                                batch_size=(batch_size if batch_size else hp_data['hp']['batch_size']))

    if torch_export:
        AV.export_to_torch(torch_export)

    if results_dict is not None:
        results_dict.update({'num_features': AV.num_features})

    return AV

def evaluator(project, outcome_label_headers, model, results_dict, input_header=None, filters=None,
              hyperparameters=None, checkpoint=None, eval_k_fold=None, max_tiles_per_slide=0, min_tiles_per_slide=0,
              normalizer=None, normalizer_source=None, batch_size=64, permutation_importance=False, histogram=False,
              save_predictions=False):

    '''Internal function to execute model evaluation process.'''

    import slideflow.model as sfmodel
    from slideflow.statistics import to_onehot

    if not isinstance(outcome_label_headers, list):
        outcome_label_headers = [outcome_label_headers]

    log.configure(filename=join(project.root, 'log.log'), verbosity=project.verbosity)

    # Load hyperparameters from saved model
    if hyperparameters:
        hp_data = sfutil.load_json(hyperparameters)
    else:
        hp_data = sfutil.get_model_hyperparameters(model)
    hp = sfmodel.HyperParameters()
    hp.load_dict(hp_data['hp'])
    model_name = f"eval-{hp_data['model_name']}-{basename(model)}"

    # Filter out slides that are blank in the outcome label, or blank in any of the input_header categories
    filter_blank = [o for o in outcome_label_headers]
    if input_header:
        input_header = [input_header] if not isinstance(input_header, list) else input_header
        filter_blank += input_header

    # Load dataset and annotations for evaluation
    eval_dataset = Dataset(config_file=project.dataset_config,
                           sources=project.datasets,
                           tile_px=hp.tile_px,
                           tile_um=hp.tile_um,
                           annotations=project.annotations,
                           filters=filters,
                           filter_blank=filter_blank)

    if hp.model_type() == 'categorical':
        if len(outcome_label_headers) == 1 and outcome_label_headers[0] not in hp_data['outcome_labels']:
            outcome_label_to_int = {outcome_label_headers[0]: {v: int(k) for k, v in hp_data['outcome_labels'].items()}}
        else:
            outcome_label_to_int = {o:
                                        { v: int(k) for k, v in hp_data['outcome_labels'][o].items() }
                                    for o in hp_data['outcome_labels']}
    else:
        outcome_label_to_int = None

    use_float = (hp.model_type() in ['linear', 'cph'])
    slide_labels_dict, unique_labels = eval_dataset.get_labels_from_annotations(outcome_label_headers,
                                                                                use_float=use_float,
                                                                                key='outcome_label',
                                                                                assigned_labels=outcome_label_to_int)

    if hp.model_type() == 'categorical' and len(outcome_label_headers) > 1:

        def process_outcome_label(v):
            return '-'.join(map(str, v)) if isinstance(v, list) else v

        labels_for_splitting = {k:{
                                    'outcome_label': process_outcome_label(v['outcome_label']),
                                    sfutil.TCGA.patient : v[sfutil.TCGA.patient]
                                  }
                                for k,v in slide_labels_dict.items()}
    else:
        labels_for_splitting = slide_labels_dict

    # If using a specific k-fold, load validation plan
    if eval_k_fold:
        log.info(f"Using {sfutil.bold('k-fold iteration ' + str(eval_k_fold))}", 1)
        validation_log = join(project.root, 'validation_plans.json')
        _, eval_tfrecords = sfio.tfrecords.get_train_and_val_tfrecords(eval_dataset,
                                                                       validation_log,
                                                                       hp.model_type(),
                                                                       labels_for_splitting,
                                                                       outcome_key='outcome_label',
                                                                       val_target=hp_data['validation_target'],
                                                                       val_strategy=hp_data['validation_strategy'],
                                                                       val_fraction=hp_data['validation_fraction'],
                                                                       val_k_fold=hp_data['validation_k_fold'],
                                                                       k_fold_iter=eval_k_fold)
    # Otherwise use all TFRecords
    else:
        eval_tfrecords = eval_dataset.get_tfrecords(merge_subdirs=True)

    # Prepare additional slide-level input
    if input_header:
        log.info('Preparing additional input', 1)
        input_header = [input_header] if not isinstance(input_header, list) else input_header
        feature_len_dict = {}   # Dict mapping input_vars to total number of different labels for each input header
        input_labels_dict = {}  # Dict mapping input_vars to nested dictionaries,
                                #    which map category ID to category label names (for categorical variables)
                                # 	or mapping to 'float' for float variables
        for slide in slide_labels_dict:
            slide_labels_dict[slide]['input'] = []

        for input_var in input_header:
            # Check if variable can be converted to float (default). If not, will assume categorical.
            try:
                eval_dataset.get_labels_from_annotations(input_var, use_float=True)
                is_float = True
            except TypeError:
                is_float = False
            log.info(f"Adding input variable {sfutil.green(input_var)} as {'float' if is_float else 'categorical'}", 1)

            if is_float:
                input_labels, _ = eval_dataset.get_labels_from_annotations(input_var, use_float=is_float)
                for slide in slide_labels_dict:
                    slide_labels_dict[slide]['input'] += input_labels[slide]['label']
                input_labels_dict[input_var] = 'float'
                feature_len_dict[input_var] = 1
            else:
                # Read categorical variable assignments from hyperparameter file
                input_label_to_int = {v: int(k) for k, v in hp_data['input_feature_labels'][input_var].items()}
                input_labels, _ = eval_dataset.get_labels_from_annotations(input_var,
                                                                           use_float=is_float,
                                                                           assigned_labels=input_label_to_int)
                feature_len_dict[input_var] = len(input_label_to_int)
                input_labels_dict[input_var] = hp_data['input_feature_labels'][input_var]

                for slide in slide_labels_dict:
                    slide_labels_dict[slide]['input'] += to_onehot(input_labels[slide]['label'],
                                                                   feature_len_dict[input_var])

        feature_sizes = [feature_len_dict[i] for i in input_header]

    else:
        input_labels_dict = None
        feature_sizes = None

    if feature_sizes and (sum(feature_sizes) != sum(hp_data['input_feature_sizes'])):
        #TODO: consider using training matrix
        raise Exception('Patient-level feature matrix not equal to what was used for model training.')
        #feature_sizes = hp_data['feature_sizes']
        #feature_names = hp_data['feature_names']
        #num_slide_features = sum(hp_data['feature_sizes'])

    # Set up model for evaluation
    # Using the project annotation file, assemble list of slides for training,
    # as well as the slide annotations dictionary (output labels)
    model_dir = join(project.models_dir, model_name)

    # Build a model using the slide list as input and the annotations dictionary as output labels
    SFM = sfmodel.SlideflowModel(model_dir,
                                 hp.tile_px,
                                 slide_labels_dict,
                                 train_tfrecords=None,
                                 validation_tfrecords=eval_tfrecords,
                                 manifest=eval_dataset.get_manifest(),
                                 mixed_precision=project.mixed_precision,
                                 model_type=hp.model_type(),
                                 normalizer=normalizer,
                                 normalizer_source=normalizer_source,
                                 feature_names=input_header,
                                 feature_sizes=feature_sizes,
                                 outcome_names=outcome_label_headers)

    # Log model settings and hyperparameters
    outcome_labels = None if hp.model_type() != 'categorical' else dict(zip(range(len(unique_labels)), unique_labels))

    hp_file = join(model_dir, 'hyperparameters.json')

    hp_data = {
        'model_name': model_name,
        'model_path': model,
        'stage': 'evaluation',
        'tile_px': hp.tile_px,
        'tile_um': hp.tile_um,
        'model_type': hp.model_type(),
        'outcome_label_headers': outcome_label_headers,
        'input_features': input_header,
        'input_feature_sizes': feature_sizes,
        'input_feature_labels': input_labels_dict,
        'outcome_labels': outcome_labels,
        'dataset_config': project.dataset_config,
        'datasets': project.datasets,
        'annotations': project.annotations,
        'validation_target': hp_data['validation_target'],
        'validation_strategy': hp_data['validation_strategy'],
        'validation_fraction': hp_data['validation_fraction'],
        'validation_k_fold': hp_data['validation_k_fold'],
        'k_fold_i': eval_k_fold,
        'filters': filters,
        'pretrain': None,
        'resume_training': None,
        'checkpoint': checkpoint,
        'hp': hp.get_dict()
    }
    sfutil.write_json(hp_data, hp_file)

    # Perform evaluation
    log.info(f'Evaluating {sfutil.bold(len(eval_tfrecords))} tfrecords', 1)

    results = SFM.evaluate(tfrecords=eval_tfrecords,
                           hp=hp,
                           model=model,
                           model_type=hp.model_type(),
                           checkpoint=checkpoint,
                           batch_size=batch_size,
                           max_tiles_per_slide=max_tiles_per_slide,
                           min_tiles_per_slide=min_tiles_per_slide,
                           permutation_importance=permutation_importance,
                           histogram=histogram,
                           save_predictions=save_predictions)

    # Load results into multiprocessing dictionary
    results_dict['results'] = results
    return results_dict

def heatmap_generator(project, slide, model_path, save_folder, roi_list, show_roi, roi_method,
                        resolution, interpolation, logit_cmap=None, vmin=0, vcenter=0.5, vmax=1,
                        buffer=True, normalizer=None, normalizer_source=None, batch_size=64, num_threads='auto'):

    '''Internal function to execute heatmap generator process.'''
    from slideflow.activations import Heatmap

    log.configure(filename=join(project.root, 'log.log'), verbosity=project.verbosity)

    resolutions = {'low': 1, 'medium': 2, 'high': 4}
    try:
        stride_div = resolutions[resolution]
    except KeyError:
        log.error(f"Invalid resolution '{resolution}': must be either 'low', 'medium', or 'high'.")
        return

    if exists(join(save_folder, f'{sfutil.path_to_name(slide)}-custom.png')):
        log.empty(f'Skipping already-completed heatmap for slide {sfutil.path_to_name(slide)}', 1)
        return

    hp_data = sfutil.get_model_hyperparameters(model_path)

    heatmap = Heatmap(slide,
                      model_path,
                      hp_data['tile_px'],
                      hp_data['tile_um'],
                      stride_div=stride_div,
                      roi_list=roi_list,
                      roi_method=roi_method,
                      buffer=buffer,
                      normalizer=normalizer,
                      normalizer_source=normalizer_source,
                      batch_size=batch_size,
                      num_threads=num_threads)

    heatmap.save(save_folder,
                 show_roi=show_roi,
                 interpolation=interpolation,
                 logit_cmap=logit_cmap,
                 vmin=vmin,
                 vcenter=vcenter,
                 vmax=vmax)

def trainer(project, outcome_label_headers, model_name, results_dict, hp, val_settings, validation_log,
            k_fold_i=None, k_fold_slide_labels=None, input_header=None, filters=None, filter_blank=None,
            pretrain=None, resume_training=None, checkpoint=None, max_tiles_per_slide=0, min_tiles_per_slide=0,
            starting_epoch=0, steps_per_epoch_override=None, normalizer=None, normalizer_source=None,
            use_tensorboard=False, multi_gpu=False, save_predictions=False, skip_metrics=False):

    '''Internal function to execute model training process.'''
    import slideflow.model as sfmodel
    import tensorflow as tf
    from slideflow.statistics import to_onehot

    log.configure(filename=join(project.root, 'log.log'), verbosity=project.verbosity)

    # First, clear prior Tensorflow graph to free memory
    tf.keras.backend.clear_session()

    # Log current model name and k-fold iteration, if applicable
    k_fold_msg = '' if not k_fold_i else f' ({val_settings.strategy} iteration {k_fold_i})'
    log.empty(f'Training model {sfutil.bold(model_name)}{k_fold_msg}...')
    log.empty(hp, 1)
    full_model_name = model_name if not k_fold_i else model_name+f'-kfold{k_fold_i}'

    # Filter out slides that are blank in the outcome label, or blank in any of the input_header categories
    if filter_blank: filter_blank += [o for o in outcome_label_headers]
    else: filter_blank = [o for o in outcome_label_headers]

    if input_header:
        input_header = [input_header] if not isinstance(input_header, list) else input_header
        filter_blank += input_header

    # Load dataset and annotations for training
    train_dts = Dataset(config_file=project.dataset_config,
                        sources=project.datasets,
                        tile_px=hp.tile_px,
                        tile_um=hp.tile_um,
                        annotations=project.annotations,
                        filters=filters,
                        filter_blank=filter_blank)

    # Load labels
    use_float = (hp.model_type() in ['linear', 'cph'])
    slide_labels_dict, unique_labels = train_dts.get_labels_from_annotations(outcome_label_headers,
                                                                             use_float=use_float,
                                                                             key='outcome_label')


    '''# ===== RNA-SEQ ===================================
    patient_to_slide = {}
    for s in slide_labels_dict:
        slide_labels_dict[s]['outcome_label'] = []
        patient = sfutil._shortname(s)
        if patient not in patient_to_slide:
            patient_to_slide[patient] = [s]
        else:
            patient_to_slide[patient] += [s]

    rna_seq_csv = '/mnt/data/TCGA_HNSC/hnsc_tcga_pan_can_atlas_2018/data_RNA_Seq_v2_mRNA_median_all_sample_Zscores.txt'
    print ('Importing csv data...')
    num_genes = 0
    with open(rna_seq_csv, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        header = next(reader)
        pt_with_rna_seq = [h[:12] for h in header[2:]]
        slide_labels_dict = {s:v for s,v in slide_labels_dict.items() if sfutil._shortname(s) in pt_with_rna_seq}
        for row in reader:
            exp_data = row[2:]
            if 'NA' in exp_data:
                continue
            num_genes += 1
            for p, exp in enumerate(exp_data):
                if pt_with_rna_seq[p] in patient_to_slide:
                    for s in patient_to_slide[pt_with_rna_seq[p]]:
                        slide_labels_dict[s]['outcome_label'] += [float(exp)]
    print(f'Loaded {num_genes} genes for {len(slide_labels_dict)} patients.')
    outcome_label_headers = None

    if True:
        outcome_labels=None

    # ========================================='''

    if hp.model_type() == 'categorical' and len(outcome_label_headers) == 1:
        outcome_labels = dict(zip(range(len(unique_labels)), unique_labels))
    elif hp.model_type() == 'categorical':
        outcome_labels = {k:dict(zip(range(len(ul)), ul)) for k, ul in unique_labels.items()}
    else:
        outcome_labels = dict(zip(range(len(outcome_label_headers)), outcome_label_headers))

    # SKIP THE BELOW IF DOING RNA-SEQ

    # If multiple categorical outcomes are used, create a merged variable for k-fold splitting
    if hp.model_type() == 'categorical' and len(outcome_label_headers) > 1:
        labels_for_splitting = {k:{
                                    'outcome_label':'-'.join(map(str, v['outcome_label'])),
                                    sfutil.TCGA.patient:v[sfutil.TCGA.patient]
                                } for k,v in slide_labels_dict.items()}
    else:
        labels_for_splitting = slide_labels_dict

    # Get TFRecords for training and validation
    manifest = train_dts.get_manifest()

    # Use external validation dataset if specified
    if val_settings.dataset:
        training_tfrecords = train_dts.get_tfrecords()
        val_dts = Dataset(tile_px=hp.tile_px,
                          tile_um=hp.tile_um,
                          config_file=project.dataset_config,
                          sources=val_settings.dataset,
                          annotations=val_settings.annotations,
                          filters=val_settings.filters,
                          filter_blank=val_settings.filter_blank)

        validation_tfrecords = val_dts.get_tfrecords()
        manifest.update(val_dts.get_manifest())
        validation_labels, _ = val_dts.get_labels_from_annotations(outcome_label_headers,
                                                                   use_float=use_float,
                                                                   key='outcome_label')
        slide_labels_dict.update(validation_labels)
    elif val_settings.strategy == 'k-fold-manual':
        all_tfrecords = train_dts.get_tfrecords()
        training_tfrecords = [tfr for tfr in all_tfrecords if k_fold_slide_labels[sfutil.path_to_name(tfr)] != k_fold_i]
        validation_tfrecords = [tfr for tfr in all_tfrecords if k_fold_slide_labels[sfutil.path_to_name(tfr)] == k_fold_i]
        num_train_str = sfutil.bold(len(training_tfrecords))
        num_val_str = sfutil.bold(len(validation_tfrecords))
        log.info(f'Using {num_train_str} TFRecords for training, {num_val_str} for validation', 1)
    else:
        tfr_split = sfio.tfrecords.get_train_and_val_tfrecords(train_dts,
                                                               validation_log,
                                                               hp.model_type(),
                                                               labels_for_splitting,
                                                               outcome_key='outcome_label',
                                                               val_target=val_settings.target,
                                                               val_strategy=val_settings.strategy,
                                                               val_fraction=val_settings.fraction,
                                                               val_k_fold=val_settings.k_fold,
                                                               k_fold_iter=k_fold_i)
        training_tfrecords, validation_tfrecords = tfr_split
    # Prepare additional slide-level input
    if input_header:
        log.info('Preparing additional input', 1)
        input_header = [input_header] if not isinstance(input_header, list) else input_header
        feature_len_dict = {} 	# Dict mapping input_vars to total number of different labels for each input header
        input_labels_dict = {}  # Dict mapping input_vars to nested dictionaries
                                #   which map category ID to category label names (for categorical variables)
                                # 	or mapping to 'float' for float variables
        for slide in slide_labels_dict:
            slide_labels_dict[slide]['input'] = []

        for input_var in input_header:

            # Check if variable can be converted to float (default). If not, will assume categorical.
            try:
                train_dts.get_labels_from_annotations(input_var, use_float=True)
                if val_settings.dataset:
                    val_dts.get_labels_from_annotations(input_var, use_float=True)
                inp_is_float = True
            except TypeError:
                inp_is_float = False
            log.info('Adding input variable ' + input_var + ' as ' + ('float' if inp_is_float else ' categorical'), 1)

            # Next, if this is a categorical variable, harmonize categories in training and validation datasets
            if (not inp_is_float) and val_settings.dataset:
                _, unique_train_input_labels = train_dts.get_labels_from_annotations(input_var, use_float=inp_is_float)
                _, unique_val_input_labels = val_dts.get_labels_from_annotations(input_var, use_float=inp_is_float)

                unique_inp_labels = sorted(list(set(unique_train_input_labels + unique_val_input_labels)))
                input_label_to_int = dict(zip(unique_inp_labels, range(len(unique_inp_labels))))
                inp_labels_dict, _ = train_dts.get_labels_from_annotations(input_var, assigned_labels=input_label_to_int)
                val_input_labels, _ = val_dts.get_labels_from_annotations(input_var, assigned_labels=input_label_to_int)
                inp_labels_dict.update(val_input_labels)
            else:
                inp_labels_dict, unique_inp_labels = train_dts.get_labels_from_annotations(input_var, use_float=inp_is_float)

            # Assign features to 'input' key of the slide-level annotations dict
            if inp_is_float:
                feature_len_dict[input_var] = num_features = 1
                for slide in slide_labels_dict:
                    slide_labels_dict[slide]['input'] += inp_labels_dict[slide]['label']
                input_labels_dict[input_var] = 'float'
            else:
                feature_len_dict[input_var] = num_features = len(unique_inp_labels)
                for slide in slide_labels_dict:
                    slide_labels_dict[slide]['input'] += to_onehot(inp_labels_dict[slide]['label'], num_features)
                input_labels_dict[input_var] = dict(zip(range(len(unique_inp_labels)), unique_inp_labels))

        feature_sizes = [feature_len_dict[i] for i in input_header]

    else:
        input_labels_dict = None
        feature_sizes = None

    # Initialize model
    # Using the project annotation file, assemble list of slides for training,
    # as well as the slide annotations dictionary (output labels)
    model_dir = join(project.models_dir, full_model_name)

    # Build a model using the slide list as input and the annotations dictionary as output labels
    SFM = sfmodel.SlideflowModel(model_dir,
                                 hp.tile_px,
                                 slide_labels_dict,
                                 training_tfrecords,
                                 validation_tfrecords,
                                 manifest=manifest,
                                 mixed_precision=project.mixed_precision,
                                 model_type=hp.model_type(),
                                 normalizer=normalizer,
                                 normalizer_source=normalizer_source,
                                 feature_names=input_header,
                                 feature_sizes=feature_sizes,
                                 outcome_names=outcome_label_headers)

    # Log model settings and hyperparameters
    hp_file = join(project.models_dir, full_model_name, 'hyperparameters.json')

    hp_data = {
        'model_name': model_name,
        'stage': 'training',
        'tile_px': hp.tile_px,
        'tile_um': hp.tile_um,
        'model_type': hp.model_type(),
        'outcome_label_headers': outcome_label_headers,
        'input_features': input_header,
        'input_feature_sizes': feature_sizes,
        'input_feature_labels': input_labels_dict,
        'outcome_labels': outcome_labels,
        'dataset_config': project.dataset_config,
        'datasets': project.datasets,
        'annotations': project.annotations,
        'validation_target': val_settings.target,
        'validation_strategy': val_settings.strategy,
        'validation_fraction': val_settings.fraction,
        'validation_k_fold': val_settings.k_fold,
        'k_fold_i': k_fold_i,
        'filters': filters,
        'pretrain': pretrain,
        'resume_training': resume_training,
        'checkpoint': checkpoint,
        'hp': hp.get_dict(),
    }
    sfutil.write_json(hp_data, hp_file)

    # Execute training
    try:
        results, history = SFM.train(hp,
                                     pretrain=pretrain,
                                     resume_training=resume_training,
                                     checkpoint=checkpoint,
                                     validate_on_batch=val_settings.validate_on_batch,
                                     val_batch_size=val_settings.batch_size,
                                     validation_steps=val_settings.validation_steps,
                                     max_tiles_per_slide=max_tiles_per_slide,
                                     min_tiles_per_slide=min_tiles_per_slide,
                                     starting_epoch=starting_epoch,
                                     steps_per_epoch_override=steps_per_epoch_override,
                                     use_tensorboard=use_tensorboard,
                                     multi_gpu=multi_gpu,
                                     save_predictions=save_predictions,
                                     skip_metrics=skip_metrics)
        results['history'] = history
        results_dict.update({full_model_name: results})
    except tf.errors.ResourceExhaustedError as e:
        log.empty('\n')
        print(e)
        log.error(f'Training failed for {sfutil.bold(model_name)}, GPU memory exceeded.', 0)
        history = None

    del SFM
    return history
