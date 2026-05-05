<!-- omit in toc -->
# Contribution Guide for quanda

**quanda** is an open source library that you can contribute to! We encourage you to contribute new metrics and explainers, optimizations or to report any bugs you encounter while using **quanda**.

In this guide, you will get a summary of the main components of **quanda**, as well as best practices for your own contributions.

If you have any questions regarding the codebase, please open an issue or write us.

## Table of Contents

- [Reporting Bugs](#reporting-bugs)
- [General Guidelines](#general-guidelines)
  - [Setting up the development environment](#setting-up-the-development-environment)
  - [Branching](#branching)
  - [Code Style](#code-style)
  - [Unit Tests](#unit-tests)
  - [Automated Checks](#automated-checks)
  - [Documentation](#documentation)
  - [Submitting a Pull Request](#submitting-a-pull-request)
- [Contributing Metrics and Benchmarks](#contributing-metrics-and-benchmarks)
  - [Contributing a New Metric](#contributing-a-new-metric)
  - [Contributing a New Benchmark](#contributing-a-new-benchmark)
- [License](#license)

## Reporting Bugs

If you come across a bug in the software, please check the repository Issues to see if this bug has already been reported. If the bug is not yet reported, please report the bug by opening an issue. Please pay attention to add a descriptive title for the bug. Briefly explain the bug in the issue body, and add details on how to reproduce the faulty behaviour whenever possible.

We will address the issue at our earliest convenience.

## General Guidelines

This section describes the prerequisites and general principles to follow while contributing to **quanda**. Please read sections [Contributing a New Metric](#contributing-a-new-metric) and [Contributing a New Benchmark](#contributing-a-new-benchmark) for implementational details.

### Setting up the development environment

Before starting to code to contribute in **quanda**, you need to install dependencies and make sure you use the correct development tools. To set up the development environment, clone the repository and install the dependencies:

```bash
pip install -e '.[dev]'
pip uninstall quanda
```

Install the pre-commit hooks to ensure code style is checked with each commit:

```bash
pre-commit install
```

Alternatively, run the makefile before a commit to ensure the code is formatted and linted correctly:

```bash
make clean-format
```

### Branching

Before you start writing your code, create a local branch from the **latest version** of `main`.

### Code Style

**quanda** follows [PEP-8](https://www.python.org/dev/peps/pep-0008/) code style. We use [ruff](https://github.com/astral-sh/ruff/) for linting and code formatting with a line-length of 79 characters.

**quanda** uses [mypy](https://mypy-lang.org/) static type checker. Please include type annotations for added code, and only write fully compatible code.

### Unit Tests

[pytest](https://github.com/pytest-dev/pytest) is used for testing.

It is possible to limit the scope of testing to specific sections of the codebase, using

```bash
pytest -m <test_marker>
```

Currently, the following markers are available to filter tests:

- utils: utils files
- explainers: Explainer wrappers
- downstream_eval_metrics: Downstream task evaluator metrics
- heuristic_metrics: Heuristic metrics
- ground_truth_metrics: Ground Truth metrics
- benchmarks: Benchmark modules
- global_ranking: global_ranking modules
- self_influence: self_influence methods of explainers
- tasks: task modules
- integration: integration tests
- slow: tests marked as slow (excluded by default; run with `pytest -m slow`)
- production_bench: production benchmark sanity checks, run only when explicitly specified

The authoritative list lives in `pytest.ini`.

Ideally, all contributions should include tests to ensure correctness.

### Automated Checks

We use `tox` for automated checks for running tests, test coverage, linting and code style. These checks are done automatically once you create a pull request, or update existing pull requests. To run them, first install tox:

```
python3 -m pip install tox
```

and then execute:

```
python3 -m tox run -e coverage
python3 -m tox run -e lint
python3 -m tox run -e type
```

### Documentation

**quanda** uses [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format for docstrings. You should add complete docstrings for contributions, as well as related references to the literature whenever possible.

### Submitting a Pull Request

Before you create your pull request, please go through this checklist to ensure a seamless review process:

- Make sure that the latest version of the code from the `main` branch is merged into your working branch.
- Run style and linting checks to format source code and detect typing errors:

```bash
make clean-format
```

- Make sure to add mypy style typing annotations whenever possible
- Create unit tests for new functionality under the `tests/` folder.
- Use `@pytest.mark` with fitting category for unit tests. If the new test cases include a new component, you can create a `@pytest.mark` category and add that category with its description to `pytest.ini`
- **quanda** strives for >90% code coverage in tests. Verify coverage and that all unit tests pass for all supported python versions by running:

```bash
python3 -m tox run -e coverage
```

Once you are done with your contributions, and have went through the above checklist:
- Create a pull request
- Provide a summary of the changes you are introducing, give details on points which might not be easily understandable.
- If the contribution is concerning an existing issue, refer to it in the body of the pull request.
- Request a review from the main contributors.

## Contributing Metrics and Benchmarks

In **quanda**, evaluation strategies are divided into 3 groups:
1. **Downstream Evaluation Tasks**: These approaches use the attributions to achieve a downstream task, like detecting mislabeled samples or predicting the class of a test sample.
2. **Heuristics**: These approaches test the attributions for desirable properties, like dependence on the model parameters and the test sample.
3. **Ground-truth**: These approaches measure the effectiveness of the attributions against a given ground truth, as in leave-1-out or leave-k-out retraining.

Each evaluation strategy has corresponding `Metric` and `Benchmark` object, and these files are organized into folders corresponding to the different kinds of evaluation strategies listed above.

In TDA evaluation, it is not uncommon to produce controlled settings (e.g. datasets that are manipulated in certain ways, while keeping track of what manipulations were exactly done, training models on these new datasets), which need to be handled with care. In **quanda**, a `Metric` object concerns itself with everything that happens in the evaluation process **after** the generation of explanations using the `Explainer` we want to evaluate. It expects to consume attributions, potentially along with extra data corresponding those attributions, to update its inner state through the `update` method. Finally, they output an overall metric score through the `compute` method.

In contrast, `Benchmark` objects concern themselves with the whole evaluation process. Each `Benchmark` object contains a `Metric` object, which it uses to compute the final score. However, `Benchmark` objects are also contain a model, a training dataset, and potentially a `Trainer` and a validation dataset.

This section goes through the different methods of `Metric` and `Benchmark` classes, with the intention of shedding light on how to structure your own contributions.

### Contributing a New Metric

To contribute a metric, first identify which group of evaluation strategies your metric belongs to and create a file for it under the directory inside the `quanda/metrics` directory. The next step is to start implementing a subclass of the base `Metric` class, defined in `quanda/metrics/base.py`. The base initializer expects the trained model and the corresponding training dataset, which all metrics that are implemented currently use. We recommend calling the base initializer in all cases.

After handling the initializations inside the `__init__` methods, the `update`, `reset` and `compute` methods should be implemented. Metrics in **quanda** are stateful. This means that they consume explanations through `update` method, and they keep record of the intermediate results of the explanations they have seen in an internal state. The `update` method should take attributions, and any extra information that is needed for the evaluation of given attributions. For example, the `ModelRandomization` metric needs to generate explanations on a randomized model, to compare with the supplied attributions. Therefore it takes also the test data which was used to generate the supplied attributions, as well as the target labels used for explaining these samples:

```python
def update(
    self,
    explanations: torch.Tensor,
    test_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
    test_targets: Optional[torch.Tensor] = None,
):
```

The first positional argument is always the attribution tensor (`explanations`); subsequent arguments are metric-specific (e.g. `test_data` / `test_targets` for metrics that need to recompute attributions on a randomized model, or `entailment_labels` for fact-tracing metrics like `MRRMetric`).

The `reset` method resets the internal state of the metric, to a state before seeing any explanations.

Finally, the `compute` method should implement generating the final score dictionary from the internal state of the metric. This dictionary should contain a key "score" and a corresponding floating point value, which is the final score of the metric. It can include additional fields that contain more information about the conducted evaluations.

These are the most important methods of the metric class. After implementing these, implement the `state_dict` and `load_state_dict` methods for the user to be able to save and restore metric states. `state_dict` should return a dictionary containing all the data needed to completely store the state of the metric, whereas `load_state_dict` should completely restore the metric state from that dictionary.

### Contributing a New Benchmark

As explained above, the `Benchmark` objects conduct the whole evaluation process, from start to finish, and use their corresponding metric to compute the final score. A benchmark in **quanda** is fully described by a YAML configuration file (see the `quanda/benchmarks/resources/configs/` directory for examples). The configuration declares the model, the training and evaluation datasets (including any wrappers such as label flipping or shortcut injection), the trainer, and any benchmark-specific options.

The base `Benchmark` class exposes four classmethods that drive a benchmark through its lifecycle:

- `train(config, ...)` — given a configuration dict, regenerate the metadata (e.g. mislabeled-sample indices, class groupings, shortcut masks), train the model, persist the checkpoint, and return a fully assembled benchmark object ready for `evaluate`.
- `from_config(config, ...)` — build a benchmark object from a configuration dict and existing assets (model checkpoint, generated metadata) without retraining.
- `load_pretrained(bench_id, cache_dir, ...)` — look up a benchmark by its registered ID in `config_map`, download the YAML / metadata / checkpoint from the Hugging Face Hub into `cache_dir`, and return the assembled benchmark.
- `train_and_push_to_hub(config, ...)` — same as `train`, plus uploading the checkpoint and the generated metadata to the Hub so the benchmark can later be `load_pretrained`-ed by anyone.

To contribute a new benchmark you generally do not need to override these four classmethods. What you should provide is:

1. **A subclass of** `Benchmark` under the appropriate `quanda/benchmarks/{downstream_eval,heuristics,ground_truth}/` subdirectory. Subclasses customize behavior via:

   - `__init__` — accept any benchmark-specific fields beyond what the base `__init__` already stores (`model`, `train_dataset`, `eval_dataset`, `checkpoints`, `checkpoints_load_func`, `device`, `val_dataset`, `use_predictions`).
   - `_extra_kwargs_from_config(cls, config, train_dataset, eval_dataset, metadata_dir, load_fresh)` — extract any subclass-specific kwargs from the YAML and return them as a dict; they get passed into `__init__` by `from_config`.
   - `_compute_and_save_indices(self, config, batch_size)` — only override if your benchmark needs to cache extra metadata on the train pass (filtered eval indices, ranking caches, etc.).
   - `evaluate(self, explainer_cls, expl_kwargs, batch_size)` — runs the explainer over `eval_dataset`, feeds the attributions to the corresponding `Metric` via `update`/`compute`, and returns the result dict (must contain `"score"`).

2. **A YAML configuration** under `quanda/benchmarks/resources/configs/` for at least one reference setup. Use existing configs (e.g. `ad1b983-default_ClassDetection.yaml`) as a template.

3. **An entry in** `config_map` so users can load your benchmark via `YourBenchmark.load_pretrained(bench_id="my_bench", ...)`.

4. **Tests** under `tests/benchmarks/` covering `from_config`, `train` (on a small unit-test config in `tests/assets/unit_bench_cfgs/`), and `evaluate`.

## License

By contributing to the project, you agree that it will be licensed under the MIT License.
