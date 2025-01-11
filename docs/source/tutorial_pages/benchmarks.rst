Benchmarks Tutorial
===================

Welcome to the benchmark tutorial of |quanda|. This tutorial walks you through the process of using the benchmarking tools in |quanda| to evaluate a data attribution method. This tutorial covers 3 different examples of benchmarks. It includes all different initialization schemes: generating the benchmark from scratch, assembling a benchmark from existing assets and downloading a precomputed benchmark.

To install the library with tutorial dependencies, run:

.. code:: bash

   pip install -e '.[tutorials]'

.. note::

   This tutorial is also available as a `notebook <https://github.com/dilyabareeva/quanda/blob/main/tutorials/demo_benchmarks.ipynb>`_.

Throughout this tutorial, we will be using a LeNet model trained on the MNIST dataset. Let's start the tutorial by importing the necessary libraries and components:

.. code-block:: python

    import os
    import sys
    import torch
    import torchvision
    from quanda.benchmarks.downstream_eval import ShortcutDetection, MislabelingDetection, SubclassDetection
    from quanda.explainers.wrappers import (
        TRAK,
        CaptumArnoldi,
        CaptumSimilarity,
        CaptumTracInCPFast,
        RepresenterPoints,
    )

Next, we need to prepare for the following computations.

.. code-block:: python

    torch.set_float32_matmul_precision("medium")
    to_img = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(mean=0.0, std=2.0),
            torchvision.transforms.Normalize(mean=-0.5, std=1.0),
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
        ]
    )

Downloading Precomputed Benchmarks
----------------------------------
In this part of the tutorial, we will use the :doc:`ShortcutDetection <../docs_api/quanda.benchmarks.downstream_eval.shortcut_detection>` metric.

We will use the benchmark corresponding to this metric to evaluate all data attributors currently included in |quanda| in terms of their ability to detect when the model is using a shortcut.

We will download the precomputed MNIST benchmark. This includes an MNIST dataset which has shortcut features (an 8-by-8 white box on a specific location) on a subset of its samples from the class 0, and a model trained on this dataset. This model has learned to classify images with these features as class 0, and we will measure the extent to which this is reflected in the attributions of different methods.

.. code-block:: python

    cache_dir = str(os.path.join(os.getcwd(), "quanda_benchmark_tutorial_cache"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark = ShortcutDetection.download(
        name="mnist_shortcut_detection",
        cache_dir=cache_dir,
        device=device,
    )

The benchmark object contains all information about the controlled evaluation setup. Run the following to get some samples with the shortcut features, using benchmark.feature_dataset and benchmark.shortcut_indices.

.. code-block:: python

    shortcut_img = benchmark.shortcut_dataset[benchmark.shortcut_indices[15]][0]
    tensor_img = torch.concat([shortcut_img, shortcut_img, shortcut_img], dim=0)
    img = to_img(tensor_img)

Prepare initialization parameters for TDA methods
+++++++++++++++++++++++++++++++++++++++++++++++++++
We now prepare the initialization parameters of attributors: hyperparameters, and components from the benchmark as needed. Note that we do not provide the model and dataset to use for attribution, since those components will be supplied by the benchmark objects, while initializing the attributor during evaluation.

- **Similarity Influence**:

.. code-block:: python

    captum_similarity_args = {
        "model_id": "mnist_shortcut_detection_tutorial",
        "layers": "model.fc_2",
        "cache_dir": os.path.join(cache_dir, "captum_similarity"),
    }
- **Arnoldi Influence Functions**: Notice that the trained checkpoints have been saved to the ``cache_dir`` while downloading the benchmark. We can reach the paths of these checkpoints with ``benchmark.get_checkpoint_paths()``.

.. code-block:: python

    hessian_num_samples = 500  # number of samples to use for hessian estimation
    hessian_ds = torch.utils.data.Subset(
        benchmark.shortcut_dataset, torch.randint(0, len(benchmark.shortcut_dataset), (hessian_num_samples,))
    )

    captum_influence_args = {
        "checkpoint": benchmark.get_checkpoint_paths()[-1],
        "layers": ["model.fc_3"],
        "batch_size": 8,
        "hessian_dataset": hessian_ds,
        "projection_dim": 5,
    }

- **TracInCP**:

.. code-block:: python

    captum_tracin_args = {
        "final_fc_layer": "model.fc_3",
        "loss_fn": torch.nn.CrossEntropyLoss(reduction="mean"),
        "checkpoints": benchmark.get_checkpoint_paths(),
        "batch_size": 8,
    }

- **TRAK**:

.. code-block:: python

    trak_args = {
        "model_id": "mnist_shortcut_detection",
        "cache_dir": os.path.join(cache_dir, "trak"),
        "batch_size": 8,
        "proj_dim": 2048,
    }

- **Representer Point Selection**:

.. code-block:: python

    representer_points_args = {
        "model_id": "mnist_shortcut_detection",
        "cache_dir": os.path.join(cache_dir, "representer_points"),
        "batch_size": 8,
        "features_layer": "model.relu_4",
        "classifier_layer": "model.fc_3",
    }

Run the benchmark evaluation on the attributors
+++++++++++++++++++++++++++++++++++++++++++++++
Note that some attributors take a long time to initialize or compute attributions. For a proof of concept, we recommend using :doc:`CaptumSimilarity <../docs_api/quanda.explainers.wrappers.captum_influence>` or :doc:`RepresenterPoints <../docs_api/quanda.explainers.wrappers.representer_points>`, or lowering the parameter values given above (i.e. using low ``proj_dim`` for TRAK or a low Hessian dataset size for :doc:`ArnoldiInfluence <../docs_api/quanda.explainers.wrappers.captum_influence>`)

.. code-block:: python

    attributors = {
        "captum_similarity": (CaptumSimilarity, captum_similarity_args),
        "captum_arnoldi" : (CaptumArnoldi, captum_influence_args),
        "captum_tracin" : (CaptumTracInCPFast, captum_tracin_args),
        "trak" : (TRAK, trak_args),
        "representer": (RepresenterPoints, representer_points_args),
    }
    results = dict()
    for name, (cls, kwargs) in attributors.items():
        results[name] = benchmark.evaluate(explainer_cls=cls, expl_kwargs=kwargs, batch_size=8)["score"]

At this point, the dictionary ``results`` contains the scores of the attributors on the benchmark.

Assembling Benchmarks from Existing Components
----------------------------------------------
You may want to handle the creation of each component differently, using different datasets, architectures, training paradigms or a higher/lower percentage of manipulated samples. We now showcase how to create and use a |quanda| :doc:`Benchmark <../docs_api/quanda.benchmarks.base>` object to use these components in the evaluation process.

To showcase different benchmarks, we will now switch to the :doc:`MislabelingDetection <../docs_api/quanda.benchmarks.downstream_eval.mislabeling_detection>` benchmark. This benchmark evaluates the ability of data atttribution methods to identify mislabeled samples in the training dataset. This is done by training a model on a dataset which has a significant number of mislabeled samples. We then use the local data attribution methods to rank the training data. Original papers propose either using self-influence (i.e. the attribution of training samples on themselves) or some special methodology for each explainer (i.e. the global coefficients of the surrogate model in Representer Points). Quanda includes efficient implementation of self-influence or other strategies proposed in the original papers, whenever possible.

This ranking is then used to go through the dataset to check mislabelings. Quanda computes the cumulative mislabeling detection curve and returns the AUC score with respect to this curve.

Instead of creating the components from scratch, we will again download the benchmark and use collect the prepared components. We will then use the ``assemble`` method to create the benchmark. Note that this is exactly what is happening when we are creating a benchmark using the ``download`` method.

.. code-block:: python

    temp_benchmark = MislabelingDetection.download(
        name="mnist_mislabeling_detection",
        cache_dir=cache_dir,
        device=device,
    )

Required Components
+++++++++++++++++++
In order to assemble a :doc:`MislabelingDetection <../docs_api/quanda.benchmarks.downstream_eval.mislabeling_detection>` benchmark, we require the following components:
- A base training dataset with correct labels.
- A dictionary containing mislabeling information: integer keys are the indices of samples to change labels, and the values correspond to the new (wrong) labels that were used to train the model
- A model trained on the mislabeled dataset
- Number of classes in the dataset
- Dataset transform that was used during training, applied to samples before feeding them to the model. If the base dataset already includes the transform, then we can just set this to ``None``, which is the case in this tutorial. If the base dataset serves raw samples, then the ``dataset_transform`` parameter allows the usage of a transform.

Let's collect these components from the downloaded benchmark. We then assemble the benchmark and evaluate the :doc:`RepresenterPoints <../docs_api/quanda.explainers.wrappers.representer_points>` attributor with it. Note that the implementation depends on computing the self-influences of the whole training dataset. This procedure is fastest for the :doc:`RepresenterPoints <../docs_api/quanda.explainers.wrappers.representer_points>` attributor. Therefore, we use this explainer here.

.. code-block:: python

    model = temp_benchmark.model
    base_dataset = temp_benchmark.base_dataset
    mislabeling_labels = temp_benchmark.mislabeling_labels
    dataset_transform = None

Assembling the benchmark and running the evaluation
++++++++++++++++++++++++++++++++++++++++++++++++++++
We are now ready to assemble and run the benchmark. After running the below code, the ``results`` dictionary will contain the score of the :doc:`RepresenterPoints <../docs_api/quanda.explainers.wrappers.representer_points>` attributor on the benchmark.

.. code-block:: python

    benchmark = MislabelingDetection.assemble(
        model=model,
        base_dataset=base_dataset,
        n_classes=10,
        mislabeling_labels=mislabeling_labels,
        dataset_transform=dataset_transform,
        device=device,
    )
    representer_points_args = {
        "model_id": "mnist_mislabeling_detection",
        "cache_dir": os.path.join(cache_dir, "representer_points"),
        "batch_size": 8,
        "features_layer": "model.relu_4",
        "classifier_layer": "model.fc_3",
    }
    results = benchmark.evaluate(
        explainer_cls=RepresenterPoints,
        expl_kwargs=representer_points_args,
    )

Generating a Benchmark from Scratch
-----------------------------------
We will now showcase how a benchmark can be created from only vanilla components. Quanda takes in all requires components and generates the benchmark, including dataset manipulations and model training, if applicable. Then the benchmark can be used to evaluate different attributors. This is done through the ``Benchmark.generate`` method.

We will go through this use-case with the :doc:`SubclassDetection <../docs_api/quanda.benchmarks.downstream_eval.subclass_detection>` benchmark which groups classes of the base dataset into superclasses. A model is trained to predict these super classes, and the original labelhighest attributed datapoint for each test sample is observed. The benchmark expects this to be the same as the true class of the test sample.

As such, we only need to provide these components to generate the benchmark:

- a model for the architecture
- a trainer: either a subclass instance of |quanda|'s :doc:`BaseTrainer <../docs_api/quanda.utils.training.trainer>` or a Lightning ``Trainer`` object. If the trainer is a Lightning trainer, the `model` has to be a Lightning module. We will use a Lightning trainer with a Lightning module.
- a base dataset
- an evaluation dataset to be used as the test set for generating the attributions to evaluate
- a dataset transform. As in the case of :doc:`MislabelingDetection <../docs_api/quanda.benchmarks.downstream_eval.mislabeling_detection>` explained above, the ``dataset_transform`` parameter can be ``None`` if the ``base_dataset`` and ``eval_dataset`` already include the required sample transformations.
- the number of superclasses we want to generate the benchmark.

Additionally, we can provide a dictionary which embodies a specific class grouping, or just use the default "random" value to randomly assign classes into superclasses, which is the approach we will take in this tutorial. Note that we will collect the base and evaluation datasets from the corresponding precomputed benchmark for simplicity and reproducibility. As such, these datasets will already include the transform required for sample normalization, which means we will supply ``dataset_transform=None``.

.. note::

    Please note that calling ``SubclassDetection.generate`` will initiate model training, therefore it will potentially take a long time.

.. code-block:: python

    from quanda.benchmarks.resources import pl_modules
    import lightning as L

    num_groups = 2
    model = pl_modules["MnistModel"](num_labels=num_groups, device=device)
    trainer = L.Trainer(max_epochs=10)
    dataset_transform = None

    # Collect base and evaluation datasets from a precomputed benchmark for simplicity, instead of creating the dataset objects from scratch
    base_dataset = temp_benchmark.base_dataset
    eval_dataset = temp_benchmark.eval_dataset


    benchmark = SubclassDetection.generate(
        model=model,
        trainer=trainer,
        base_dataset=base_dataset,
        eval_dataset=eval_dataset,
        dataset_transform=dataset_transform,
        n_classes=10,
        n_groups=num_groups,
        class_to_group="random",
    )

Now that we have trained the model on the MNIST dataset with randomly grouped classes, we finalize this tutorial by evaluating the :doc:`CaptumSimilarity <../docs_api/quanda.explainers.wrappers.captum_influence>` attributor. The ``results`` dictionary will contain the score of the attributor on the benchmark after running the following:

.. code-block:: python

    results = benchmark.evaluate(
        explainer_cls=CaptumSimilarity,
        expl_kwargs={
            "model_id": "mnist_subclass_detection_tutorial",
            "layers": "model.fc_2",
            "cache_dir": os.path.join(cache_dir, "captum_similarity"),
        },
    )
