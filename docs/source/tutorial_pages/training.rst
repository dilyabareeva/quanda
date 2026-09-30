Training Configuration
======================

Several |quanda| components retrain models as part of the evaluation
protocol:

- The :doc:`LinearDatamodelingMetric <../docs_api/quanda.metrics.ground_truth.linear_datamodeling>`
  and the corresponding
  :doc:`LinearDatamodeling <../docs_api/quanda.benchmarks.ground_truth.linear_datamodeling>`
  benchmark retrain the model on ``m`` random subsets of the training
  data.
- The ``train`` method of the
  :doc:`Benchmark <../docs_api/quanda.benchmarks.base>` classes trains
  the benchmark's main model from scratch on the (modified) training
  dataset.

Training in |quanda| is handled through trainer instances passed to
these components. Components that retrain models accept either a
standard `lightning.Trainer
<https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_ or an
implementation of the abstract
:doc:`BaseTrainer <../docs_api/quanda.utils.training.trainer>`
interface, of which |quanda| ships a minimal implementation
(``quanda.utils.training.Trainer``). In practice, this means that the
training configuration is not constrained: a trainer can implement
arbitrary training configurations, including multi-node and multi-GPU
setups.

The trainer interface
---------------------

``BaseTrainer`` requires a single method, ``fit``, which receives the
model and the dataloaders and returns the trained model:

.. code-block:: python

    from typing import List, Optional

    import lightning as L
    import torch

    from quanda.utils.training import BaseTrainer


    class MyTrainer(BaseTrainer):
        def fit(
            self,
            model: torch.nn.Module,
            train_dataloaders: torch.utils.data.DataLoader,
            val_dataloaders: Optional[torch.utils.data.DataLoader] = None,
            accelerator: str = "cpu",
            devices: int = 0,
            seed: int = 42,
            callbacks: Optional[List[L.Callback]] = None,
            trainer_fit_kwargs: Optional[dict] = None,
            *args,
            **kwargs,
        ) -> torch.nn.Module:
            ...

Inside ``fit`` you are free to run any training procedure: a plain
PyTorch loop, a Hugging Face ``Trainer``, an ``accelerate``-based
setup, or a distributed multi-node job — as long as the trained
``torch.nn.Module`` is returned.

Using the built-in Trainer
--------------------------

``quanda.utils.training.Trainer`` is a minimal ``BaseTrainer``
implementation that wraps a ``lightning.Trainer`` around a plain
``torch.nn.Module``. It is configured through standard training
hyperparameters:

.. code-block:: python

    import torch

    from quanda.metrics.ground_truth import LinearDatamodelingMetric
    from quanda.utils.training import Trainer

    trainer = Trainer(
        optimizer=torch.optim.Adam,
        lr=0.001,
        max_epochs=100,
        criterion=torch.nn.CrossEntropyLoss(),
        scheduler=torch.optim.lr_scheduler.ConstantLR,
        optimizer_kwargs={"weight_decay": 0.0},
        seed=42,
    )

    metric = LinearDatamodelingMetric(
        model=model,
        train_dataset=train_dataset,
        trainer=trainer,
        alpha=0.5,
        m=100,
        cache_dir="./cache",
        model_id="lds_demo",
    )

Using a lightning.Trainer
-------------------------

If your model is a ``lightning.LightningModule``, you can instead pass
a ``lightning.Trainer`` directly. This gives you access to the full
Lightning feature set — distributed strategies, mixed precision,
gradient accumulation, custom callbacks and loggers:

.. code-block:: python

    import lightning as L

    from quanda.metrics.ground_truth import LinearDatamodelingMetric

    trainer = L.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=4,                # multi-GPU training
        strategy="ddp",           # or "fsdp", "deepspeed", ...
        precision="16-mixed",
    )

    metric = LinearDatamodelingMetric(
        model=lightning_module,   # must be a LightningModule
        train_dataset=train_dataset,
        trainer=trainer,
        alpha=0.5,
        m=100,
        cache_dir="./cache",
        model_id="lds_demo",
    )

Note that when a ``lightning.Trainer`` is used, the model passed to the
component must be a ``lightning.LightningModule``; when a
``BaseTrainer`` is used, it must be a plain ``torch.nn.Module``.

Writing a custom BaseTrainer
----------------------------

The example below implements a self-contained PyTorch training loop.
Any training configuration can be expressed this way:

.. code-block:: python

    from typing import List, Optional

    import lightning as L
    import torch

    from quanda.utils.training import BaseTrainer


    class SGDTrainer(BaseTrainer):
        def __init__(self, lr: float = 0.01, max_epochs: int = 10):
            self.lr = lr
            self.max_epochs = max_epochs

        def fit(
            self,
            model: torch.nn.Module,
            train_dataloaders: torch.utils.data.DataLoader,
            val_dataloaders: Optional[torch.utils.data.DataLoader] = None,
            accelerator: str = "cpu",
            devices: int = 0,
            seed: int = 42,
            callbacks: Optional[List[L.Callback]] = None,
            trainer_fit_kwargs: Optional[dict] = None,
            *args,
            **kwargs,
        ) -> torch.nn.Module:
            torch.manual_seed(seed)
            device = "cuda" if accelerator == "gpu" else "cpu"
            model = model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
            criterion = torch.nn.CrossEntropyLoss()

            model.train()
            for epoch in range(self.max_epochs):
                for inputs, targets in train_dataloaders:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), targets)
                    loss.backward()
                    optimizer.step()
            return model

Trainer configuration in benchmark YAML configs
-----------------------------------------------

The ``train`` methods of the pre-computed benchmarks are driven by YAML
configuration files (see the shipped configs in
``quanda/benchmarks/resources/configs``). A trainer instance can be
passed directly via the ``trainer`` argument of ``train``:

.. code-block:: python

    benchmark = ClassDetection.train(config, trainer=my_trainer)

If no trainer is passed, the ``trainer`` section of the model
configuration is parsed into a built-in ``Trainer``:

.. code-block:: yaml

    model:
      trainer:
        optimizer: adam
        optimizer_kwargs:
          weight_decay: 0.0
        criterion: cross_entropy
        scheduler: constant
        scheduler_kwargs:
          last_epoch: -1
        lr: 0.001
        max_epochs: 200

The registered names are mapped in
``quanda/utils/training/options.py``:

.. list-table::
  :header-rows: 1

  * - Field
    - Registered names
  * - ``optimizer``
    - ``sgd``, ``adam``, ``adamw``, ``rmsprop``
  * - ``criterion``
    - ``mse``, ``cross_entropy``, ``bce``, ``bce_with_logits``, ``l1``,
      ``smooth_l1``, ``nll``
  * - ``scheduler``
    - ``step``, ``cosine``, ``reduce_on_plateau``, ``one_cycle``,
      ``constant``, ``linear``

Optional fields with defaults: ``seed`` (42), ``num_workers`` (0),
``enable_progress_bar`` (true) and ``gradient_clip_val`` (none).
