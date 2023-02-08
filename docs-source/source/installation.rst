Installation
============

Slideflow has been tested and is supported on the following systems:

- Ubuntu 18.04, 20.04, and 22.04
- Centos 7, 8, and 8 Stream

Requirements
************

- Python >= 3.7 (<3.10 if using `cuCIM <https://docs.rapids.ai/api/cucim/stable/>`_)
- `Tensorflow <https://www.tensorflow.org/>`_ 2.5-2.9 *or* `PyTorch <https://pytorch.org/>`_ 1.9-1.12

Optional
--------

- `Libvips >= 8.9 <https://libvips.github.io/libvips/>`_ (alternative slide reader, adds support for \*.scn, \*.mrxs, \*.ndpi, \*.vms, and \*.vmu files)
- `QuPath <https://qupath.github.io>`_ (for pathologist ROIs)
- Linear solver (for site-preserved cross-validation):

  - `CPLEX 20.1.0 <https://www.ibm.com/docs/en/icos/12.10.0?topic=v12100-installing-cplex-optimization-studio>`_ with `Python API <https://www.ibm.com/docs/en/icos/12.10.0?topic=cplex-setting-up-python-api>`_
  - *or* `Pyomo <http://www.pyomo.org/installation>`_ with `Bonmin <https://anaconda.org/conda-forge/coinbonmin>`_ solver

Download with pip
*****************

Slideflow can be installed either with PyPI or as a Docker container. To install via pip:

.. code-block:: bash

    # Update to latest pip
    pip install --upgrade pip wheel

    # Current stable release, Tensorflow backend
    pip install slideflow[tf] cucim cupy-cuda11x

    # Alternatively, install with PyTorch backend
    pip install slideflow[torch] cucim cupy-cuda11x

The ``cupy`` package name depends on the installed CUDA version; `see here <https://docs.cupy.dev/en/stable/install.html#installing-cupy>`_ for installation instructions. ``cucim`` and ``cupy`` are not required if using Libvips.

Run a Docker container
**********************

Alternatively, pre-configured `docker images <https://hub.docker.com/repository/docker/jamesdolezal/slideflow>`_ are available with cuCIM, Libvips, and either PyTorch 1.11 or Tensorflow 2.8 pre-installed. Using a preconfigured `Docker <https://docs.docker.com/install/>`_ container is the easiest way to get started with compatible dependencies and GPU support.

To run a Docker container with the Tensorflow 2.8 backend:

.. code-block:: bash

    docker pull jamesdolezal/slideflow:latest-tf
    docker run -it --gpus all jamesdolezal/slideflow:latest-tf

To run a Docker container with the PyTorch 1.11 backend:

.. code-block:: bash

    docker pull jamesdolezal/slideflow:latest-torch
    docker run -it --shm-size=2g --gpus all jamesdolezal/slideflow:latest-torch

Build from source
*****************

To build Slideflow from source, clone the repository from the project `Github page <https://github.com/jamesdolezal/slideflow>`_:

.. code-block:: bash

    git clone https://github.com/jamesdolezal/slideflow
    git submodule init && git submodule update --recursive
    cd slideflow
    conda env create -f environment.yml
    conda activate slideflow
    python setup.py bdist_wheel
    pip install dist/slideflow* cupy-cuda11x

.. warning::
    A bug in the pixman library (version=0.38) will corrupt downsampled slide images, resulting in large black boxes across the slide. We have provided a patch for version 0.38 that has been tested for Ubuntu, which is provided in the project `Github page <https://github.com/jamesdolezal/slideflow>`_ (``pixman_repair.sh``), although it may not be suitable for all environments and we make no guarantees regarding its use. The `Slideflow docker images <https://hub.docker.com/repository/docker/jamesdolezal/slideflow>`_ already have this applied. If you are installing from source, have pixman version 0.38, and are unable to apply this patch, the use of downsampled image layers must be disabled to avoid corruption (pass ``enable_downsample=False`` to tile extraction functions).

Tensorflow vs. PyTorch
**********************

Slideflow supports both Tensorflow and PyTorch, with cross-compatible TFRecord storage. Slideflow will default to using Tensorflow if both are available, but the backend can be manually specified using the environmental variable ``SF_BACKEND``. For example:

.. code-block:: console

    export SF_BACKEND=torch

.. _slide_backend:

cuCIM vs. Libvips
*****************

By default, Slideflow reads whole-slide images using `cuCIM <https://docs.rapids.ai/api/cucim/stable/>`_. Although much faster than other openslide-based frameworks, it supports fewer slide scanner formats. Slideflow also includes a `Libvips <https://libvips.github.io/libvips/>`_ backend, which adds support for \*.scn, \*.mrxs, \*.ndpi, \*.vms, and \*.vmu files. You can set the active slide backend with the environmental variable ``SF_SLIDE_BACKEND``:

.. code-block:: console

    export SF_SLIDE_BACKEND=libvips
