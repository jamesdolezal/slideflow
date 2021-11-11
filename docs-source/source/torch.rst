PyTorch
=======

While the primary backend for this package is Tensorflow/Keras, it includes a PyTorch TFRecord reader to ensure extracted and saved image tiles can be served to either Tensorflow or PyTorch models in cross-compatible fashion. There is also early support for a PyTorch model Trainer; full backend support is in development.


TFRecord DataLoader
*******************

To use existing TFRecords that contain tiles extracted with ``slideflow``, use the :class:`slideflow.dataset.Dataset` class and its :func:`torch` function to create a DataLoader instance that loads and interleaves TFRecords with the same functionality as the Tensorflow backend. To start, load a saved project and select a dataset:

.. code-block:: python

    P = sf.Project('/project/path', ...)
    dts = P.dataset(tile_px=299, tile_um=302, filters=None)

If you want to perform any balancing, use the :meth:`slideflow.dataset.Datset.balance` method:

.. code-block:: python

    dts = dts.balance('HPV_status', strategy='category')

Finally, use the :meth:`slideflow.dataset.Dataset.torch` method to create a DataLoader object:

.. code-block:: python

    dataloader = dts.torch(
        labels       = ...       # Your outcome label
        batch_size   = 64,       # Batch size
        num_workers  = 6,        # Number of workers reading tfrecords
        infinite     = True,     # True for training, False for validation
        augment      = True,     # Flip/rotate/compression augmentation
        standardize  = True,     # Standardize images: mean 0, variance of 1
        pin_memory   = False,    # Pin memory to GPUs
    )

The returned dataloader can then be used directly with your PyTorch applications.

Training with PyTorch
*********************

Single-outcome categorical models can be trained with the experimental PyTorch backend by setting the environmental
variable ``SF_BACKEND`` to "torch". This can be done with either ``os.environ``, or by calling
``slideflow.set_backend('torch')``. Models are then trained in the same way, using :func:`sf.Project.train`.
Some options are not yet implemented, and models other than single-outcome categorical models are not yet supported.