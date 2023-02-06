.. currentmodule:: slideflow.model

slideflow.model
===============

This module provides the :class:`ModelParams` class to organize model and training
parameters/hyperparameters and assist with model building, as well as the :class:`Trainer` class that
executes model training and evaluation. :class:`LinearTrainer` and :class:`CPHTrainer`
are extensions of this class, supporting linear and Cox Proportional Hazards outcomes, respectively. The function
:func:`build_trainer` can choose and return the correct model instance based on the provided
hyperparameters.

.. note::
    In order to support both Tensorflow and PyTorch backends, the :mod:`slideflow.model` module will import either
    :mod:`slideflow.model.tensorflow` or :mod:`slideflow.model.torch` according to the currently active backend,
    indicated by the environmental variable ``SF_BACKEND``.

See :ref:`training` for a detailed look at how to train models

Mini-batch balancing
********************

During training, mini-batch balancing can be customized to assist with increasing representation of sparse outcomes or small slides. Five mini-batch balancing methods are available when configuring :class:`slideflow.ModelParams`, set through the parameters ``training_balance`` and ``validation_balance``. These are ``'tile'``, ``'category'``, ``'patient'``, ``'slide'``, and ``'none'``.

If **tile-level balancing** ("tile") is used, tiles will be selected randomly from the population of all extracted tiles.

If **slide-based balancing** ("patient") is used, batches will contain equal representation of images from each slide.

If **patient-based balancing** ("patient") is used, batches will balance image tiles across patients. The balancing is similar to slide-based balancing, except across patients (as each patient may have more than one slide).

If **category-based balancing** ("category") is used, batches will contain equal representation from each outcome category.

If **no balancing** is performed, batches will be assembled by randomly selecting from TFRecords. This is equivalent to slide-based balancing if each slide has its own TFRecord (default behavior).

If you are :ref:`using a Trainer <training_with_trainer>` to train your models, you can further customize the mini-batch balancing strategy by using :meth:`slideflow.Dataset.balance` on your training and/or validation datasets.

ModelParams
***********
.. autoclass:: ModelParams
.. autofunction:: slideflow.ModelParams.to_dict
.. autofunction:: slideflow.ModelParams.get_normalizer
.. autofunction:: slideflow.ModelParams.validate
.. autofunction:: slideflow.ModelParams.model_type

Trainer
*******
.. autoclass:: Trainer
.. autofunction:: slideflow.model.Trainer.load
.. autofunction:: slideflow.model.Trainer.evaluate
.. autofunction:: slideflow.model.Trainer.predict
.. autofunction:: slideflow.model.Trainer.train

LinearTrainer
*************
.. autoclass:: LinearTrainer

CPHTrainer
***********
.. autoclass:: CPHTrainer

Features
********
.. autoclass:: Features
.. autofunction:: slideflow.model.Features.from_model
.. autofunction:: slideflow.model.Features.__call__

Other functions
***************
.. autofunction:: build_trainer
.. autofunction:: is_tensorflow_model
.. autofunction:: is_tensorflow_tensor
.. autofunction:: is_torch_model
.. autofunction:: is_torch_tensor
.. autofunction:: read_hp_sweep