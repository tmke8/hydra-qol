hydra-qol
=========

Quality-of-life helpers for `Hydra <https://hydra.cc>`_.

Hydra's `structured configs <https://hydra.cc/docs/tutorials/structured_config/intro/>`_
let you define a project's configuration as Python dataclasses, with YAML files on top
for the configurations that you use often. ``hydra-qol`` is a small collection of
functions that make this setup smoother:

* :func:`~hydra_qol.register_hydra_config` registers the main config class and all
  config-group options in Hydra's ``ConfigStore`` in one call, and checks the config
  classes for mistakes that Hydra would otherwise only report with cryptic errors.
* :func:`~hydra_qol.prepare_for_logging` turns a composed config into a plain
  dictionary that is nice to log (e.g. with Weights & Biases), with the chosen
  config-group options visible in the keys.
* :func:`~hydra_qol.check_configs` composes *every* YAML file in your config directory
  and instantiates the config object from it, so that renamed fields, typos and wrong
  value types in YAML files are caught in CI instead of at the start of an experiment.

A complete, annotated example project that uses these functions can be found at
`wearepal/hydra-example <https://github.com/wearepal/hydra-example>`_.

Installation
------------

.. code-block:: bash

   uv add hydra-qol
   # or
   pip install hydra-qol

``hydra-qol`` requires Python 3.11 or newer and depends only on ``hydra-core`` (1.4 or
newer; at the time of writing that is a pre-release) and ``omegaconf`` (2.4 or newer).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   usage
   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
