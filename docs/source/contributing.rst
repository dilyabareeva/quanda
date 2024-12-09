Contribution Guide for |quanda|
=============================

|quanda| is an open source library that you can contribute to! We
encourage you to contribute new metrics and explainers, optimizations or
to report any bugs you encounter while using |quanda|.

In this guide, you will get a summary of the main components of
|quanda|, as well as best practices for your own contributions.

If you have any questions regarding the codebase, please `open an
issue <https://github.com/dilyabareeva/quanda/issues/new/choose>`__ or write us
at dilyabareeva@gmail.com or galip.uemit.yolcu@hhi.fraunhofer.de.

Table of Contents
-----------------

-  `Reporting Bugs <#reporting-bugs>`__
-  `General Guidelines <#general-guidelines>`__
-  `Setting Up the Development
   Environment <#setting-up-the-development-environment>`__
-  `Branching <#branching>`__
-  `Code Style <#code-style>`__
-  `Unit Tests <#unit-tests>`__
-  `Automated Checks <#automated-checks>`__
-  `Documentation <#documentation>`__
-  `Submitting a Pull Request <#submitting-a-pull-request>`__
-  `Contributing Metrics and
   Benchmarks <#contributing-metrics-and-benchmarks>`__
-  `Contributing a New Metric <#contributing-a-new-metric>`__
-  `Contributing a New Benchmark <#contributing-a-new-benchmark>`__
-  `Caveats and Pitfalls <#caveats-and-pitfalls>`__
-  `License <#license>`__

Reporting Bugs
--------------

If you come across a bug in the software, please check the repository
`Issues <https://github.com/dilyabareeva/quanda/issues>`__ to see if
this bug has already been reported. If the bug is not yet reported,
please report the bug by `opening an
issue <https://github.com/dilyabareeva/quanda/issues/new>`__. Please pay
attention to add a descriptive title for the bug. Briefly explain
the bug in the issue body, and add details on how to reproduce the faulty
behaviour whenever possible.

We will address the issue at our earliest convenience.

General Guidelines
------------------

This section describes the prerequisites and general principles to
follow while contributing to |quanda|. Please read sections
`Contributing a New Metric <#contributing-a-new-metric>`__ and
`Contributing a New Benchmark <#contributing-a-new-benchmark>`__ for
implementational details.

Setting up the development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting to code to contribute in |quanda|, you need to install
dependencies and make sure you use the correct development tools. To set
up the development environment, clone the repository and install the
dependencies:

.. code:: bash

   pip install -e '.[dev]'
   pip uninstall quanda

Install the pre-commit hooks to ensure code style is checked with each
commit:

.. code:: bash

   pre-commit install

Alternatively, run the makefile before a commit to ensure the code is
formatted and linted correctly:

.. code:: bash

   make clean-format

Branching
~~~~~~~~~

Before you start writing your code, create a local branch from the
**latest version** of ``main``.

Code Style
~~~~~~~~~~

|quanda| follows `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`__
code style. We use `ruff <https://github.com/astral-sh/ruff/>`__ for
linting and code formatting with a line-length of 79 characters.


|quanda| uses `mypy <https://mypy-lang.org/>`__ static type checker.
Please include type annotations for added code, and only write fully
compatible code.

Unit Tests
~~~~~~~~~~

`pytest <https://github.com/pytest-dev/pytest>`__ is used for testing.

It is possible to limit the scope of testing to specific sections of the
codebase, using

.. code:: bash

   pytest -m <test_marker>

Currently, the following markers are available to filter tests:

-  utils: utils files
-  explainers: Explainer wrappers
-  downstream_eval_metrics: Downstream task evaluator metrics
-  heuristic_metrics: Heuristic metrics
-  ground_truth_metrics: Ground Truth metrics
-  benchmarks: Benchmark modules
-  aggregators: Aggregator modules
-  aggr_strategies: aggr_strategies modules
-  self_influence: self_influence methods of explainers

Ideally, all contributions should include tests to ensure correctness.

Automated Checks
~~~~~~~~~~~~~~~~

We use ``tox`` for automated checks for running tests, test coverage,
linting and code style. These checks are done automatically once you
create a pull request, or update existing pull requests. To run them,
first install tox:

::

   python3 -m pip install tox

and then execute:

::

   python3 -m tox run -e coverage
   python3 -m tox run -e lint
   python3 -m tox run -e type

Documentation
~~~~~~~~~~~~~

|quanda| uses
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`__
format for docstrings. You should add complete docstrings for
contributions, as well as related references to the literature whenever
possible.

Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~

Before you create your pull request, please go through this checklist to
ensure a seamless review process:

-  Make sure that the latest version of the code from the ``main``
   branch is merged into your working branch.
-  Run style and linting checks to format source code and detect typing
   errors:

.. code:: bash

   make clean-format

-  Make sure to add mypy style typing annotations whenever possible
-  Create unit tests for new functionality under the ``tests/`` folder.
-  Use ``@pytest.mark`` with fitting category for unit tests. If the new
   test cases include a new component, you can create a ``@pytest.mark``
   category and add that category with its description to ``pytest.ini``
-  |quanda| strives for >90% code coverage in tests. Verify coverage
   and that all unit tests pass for all supported python versions by
   running:

.. code:: bash

   python3 -m tox run -e coverage

Once you are done with your contributions, and have went through the
above checklist: - Create a `pull
request <https://github.com/dilyabareeva/quanda/compare>`__ - Provide a
summary of the changes you are introducing, give details on points which
might not be easily understandable. - If the contribution is concerning
an existing issue, refer to it in the body of the pull request. -
Request a review from `dilyabareeva <https://github.com/dilyabareeva>`__
or `gumityolcu <https://github.com/gumityolcu>`__.

Contributing Metrics and Benchmarks
-----------------------------------

In |quanda|, evaluation strategies are divided into 3 groups:
1-\ **Downstream Evaluation Tasks**: These approaches use the
attributions to achieve a downstream task, like detecting mislabeled
samples or predicting the class of a test sample. 2-\ **Heuristics**:
These approaches test the attributions for desirable properties, like
dependence on the model parameters and the test sample.
3-\ **Ground-truth**: These approaches measure the effectiveness of the
attributions against a given ground truth, as in leave-1-out or
leave-k-out retraining.

Each evaluation strategy has corresponding :doc:`Metric <docs_api/quanda.metrics.base>` and :doc:`Benchmark <docs_api/quanda.benchmarks.base>`
object, and these files are organized into folders corresponding to the
different kinds of evaluation strategies listed above.

In TDA evaluation, it is not uncommon to produce controlled settings
(e.g. datasets that are manipulated in certain ways, while keeping track
of what manipulations were exactly done, training models on these new
datasets), which need to be handled with care. In |quanda|, a
:doc:`Metric <docs_api/quanda.metrics.base>` object concerns itself with everything that happens in the
evaluation process **after** the generation of explanations using the
:doc:`Explainer <docs_api/quanda.explainers.base>` we want to evaluate. It expects to consume attributions,
potentially along with extra data corresponding those attributions, to
update its inner state through the ``update`` method. Finally, they
output an overall metric score through the ``compute`` method.

In contrast, :doc:`Benchmark <docs_api/quanda.benchmarks.base>` objects concern themselves with the whole
evaluation process. Each :doc:`Benchmark <docs_api/quanda.benchmarks.base>` object contains a :doc:`Metric <docs_api/quanda.metrics.base>`
object, which it uses to compute the final score. However, :doc:`Benchmark <docs_api/quanda.benchmarks.base>`
objects are also contain a model, a training dataset, and potentially a
``Trainer`` and a validation dataset.

This section goes through the different methods of :doc:`Metric <docs_api/quanda.metrics.base>` and
:doc:`Benchmark <docs_api/quanda.benchmarks.base>` classes, with the intention of shedding light on how to
structure your own contributions.

Contributing a New Metric
~~~~~~~~~~~~~~~~~~~~~~~~~

To contribute a metric, first identify which group of evaluation
strategies your metric belongs to and create a file for it under the
directory inside the ``quanda/metrics`` directory. The next step is to
start implementing a subclass of the base :doc:`Metric <docs_api/quanda.metrics.base>` class, defined in
``quanda/metrics/base.py``. The base initializer expects the trained
model and the corresponding training dataset, which all metrics that are
implemented currently use. We recommend calling the base initializer in
all cases.

After handling the initializations inside the ``__init__`` methods, the
``update``, ``reset`` and ``compute`` methods should be implemented.
Metrics in |quanda| are stateful. This means that they consume
explanations through ``update`` method, and they keep record of the
intermediate results of the explanations they have seen in an internal
state. The ``update`` method should take attributions, and any extra
information that is needed for the evaluation of given attributions. For
example, the ``ModelRandomization`` metric needs to generate
explanations on a randomized model, to compare with the supplied
attributions. Therefore it takes also the test data which was used to
generate the supplied attributions, as well as the target labels used
for explaining these samples:

::

       def update(
           self,
           test_data: torch.Tensor,
           explanations: torch.Tensor,
           explanation_targets: Optional[torch.Tensor] = None,
       ):

The ``reset`` method resets the internal state of the metric, to a state
before seeing any explanations.

Finally, the ``compute`` method should implement generating the final
score dictionary from the internal state of the metric. This dictionary
should contain a key “score” and a corresponding floating point value,
which is the final score of the metric. It can include additional fields
that contain more information about the conducted evaluations.

These are the most important methods of the metric class. After
implementing these, implement the ``state_dict`` and ``load_state_dict``
methods for the user to be able to save and restore metric states.
``state_dict`` should return a dictionary containing all the data needed
to completely store the state of the metric, whereas ``load_state_dict``
should completely restore the metric state from that dictionary. ###
Contributing a New Benchmark As explained above, the :doc:`Benchmark <docs_api/quanda.benchmarks.base>`
objects conduct the whole evaluation process, from start to finish.
Thus, they use their corresponding metric. Benchmarks are not
initialized using the ``__init__`` method. Instead, |quanda| offers
different initialization strategies. Below, we list the initialization
methods that you should implement, along with their functionalities:

The class method ``generate`` accepts a trained ``model`` to be
explained, a vanilla ``train_dataset`` to be used, and other components
required by the benchmark to run the evaluation process from start to
finish. The ``train_dataset`` should have type annotation
``Union[str, torch.utils.data.Dataset]``, since we want to allow for a
downloadable benchmark using a HuggingFace dataset, which we take from
the user as a string. Another input, ``dataset_split : str = "train"``
is also needed, to use when a HuggingFace dataset is downloaded. When
you are implementing the ``generate`` function, you should additionally:
- Create an instance of the :doc:`Benchmark <docs_api/quanda.benchmarks.base>` to return:

::

   obj = cls()

-  Infer device from the passed model using the base method:

::

   obj._set_devices(model)

-  Populate ``train_dataset`` field of ``obj``:

::

   obj.train_dataset = obj._process_dataset(train_dataset, dataset_split)

-  Populate the rest of the required fields of the ``obj`` object from
   the parameters of the method.
-  If the benchmark requires training a model on a modified dataset,
   ``generate`` should take a ``BaseTrainer`` or a Lightning ``Trainer``
   object as a parameter and handle the training.

The class method ``assemble`` should generate the :doc:`Benchmark <docs_api/quanda.benchmarks.base>` object
from existing components, generated beforehand with the ``generate``
method. Again, it should take a ``train_dataset`` and ``model``. You
should again: - Create an instance of the :doc:`Benchmark <docs_api/quanda.benchmarks.base>` to return:

::

   obj = cls()

-  Infer device from the passed model using the base method:

::

   obj._set_devices(model)

-  Populate ``train_dataset`` field of ``obj``:

::

   obj.train_dataset = obj._process_dataset(train_dataset, dataset_split)

-  Populate the rest of the required fields of the ``obj`` object from
   the parameters of the method.
-  If the benchmark requires training a model, the ``model`` should be a
   model trained already in the correct context. This constitutes the
   main difference between the ``generate`` and ``assemble`` methods.
   Thus, ``assemble`` is used to skip the costly training process.
   Otherwise, the ``assemble`` method is generally the same as the
   ``generate`` method.

Finally, the class method ``download`` is needed to download and
assemble a benchmark from precomputed component. We will handle this
method once your pull request is reviewed and merged.

License
-------

By contributing to the project, you agree that it will be licensed under
the MIT License.
