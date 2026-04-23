"""Shared base for fact-tracing benchmarks (MRR, Recall@k, Tail-Patch).

Lifts the three concerns they share — config-driven construction,
prompt/evidence/entailment dataset loading, and the evaluate() loop
skeleton — into a single class so the concrete benchmarks only need
to declare their metric and metric-specific kwargs.
"""

from typing import Any, Callable, List, Optional, Union

import datasets  # type: ignore
import torch

from quanda.benchmarks.base import Benchmark, _resolve_ckpts
from quanda.benchmarks.config_parser import (
    BenchConfigParser,
    FactTracingConfigParser,
)
from quanda.metrics import Metric


class FactTracingBenchmark(Benchmark):
    """Base class for fact-tracing benchmarks.

    Subclasses provide:
    - ``name`` class attribute (benchmark display name)
    - ``eval_args`` class attribute (metric input keys)
    - ``_build_metric(self)`` returning the concrete metric instance
    - ``_extra_kwargs_from_config(cls, config)`` (optional) for extra
      ``__init__`` kwargs beyond ``entailment_labels``
    """

    name: str
    eval_args: list = ["explanations", "entailment_labels"]

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        checkpoints: List[str],
        checkpoints_load_func: Callable[..., Any],
        device: str = "cpu",
        val_dataset: Optional[
            Union[torch.utils.data.Dataset, datasets.Dataset]
        ] = None,
        use_predictions: bool = False,
        entailment_labels: Optional[torch.Tensor] = None,
    ):
        """Mirror :class:`~quanda.benchmarks.base.Benchmark`'s signature.

        Adds ``entailment_labels`` — the binary ``(n_eval, n_train)`` fact
        matrix — as a fact-tracing-specific attribute.
        """
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            device=device,
            val_dataset=val_dataset,
            use_predictions=use_predictions,
        )
        self.entailment_labels: Optional[torch.Tensor] = entailment_labels

    @classmethod
    def from_config(
        cls,
        config: dict,
        load_meta_from_disk: bool = True,
        offline: bool = False,
        device: str = "cpu",
        metadata_suffix: str = "",
        load_fresh: bool = False,
    ) -> "FactTracingBenchmark":
        """Build the benchmark from a YAML-derived config dict.

        Loads prompts/evidence/entailment via
        :func:`load_fact_tracing_datasets_from_cfg` (which bypasses the
        generic dataset parser because one HF dataset fans out into
        both splits) and the model via the standard
        :class:`BenchConfigParser` path.
        """
        if offline and load_fresh:
            raise ValueError(
                "offline=True and load_fresh=True are incompatible."
            )

        prompt_ds, evidence_ds, entailment_labels, _ = (
            FactTracingConfigParser.parse_fact_tracing_cfg(
                config["fact_tracing"]
            )
        )

        model, checkpoints, checkpoints_load_func = (
            BenchConfigParser.parse_model_cfg(
                model_cfg=config["model"],
                bench_save_dir=config.get("bench_save_dir", "./tmp"),
                ckpts=_resolve_ckpts(config),
                offline=offline,
                load_fresh=load_fresh,
                device=device,
            )
        )

        return cls(
            model=model,
            train_dataset=evidence_ds,
            eval_dataset=prompt_ds,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            device=device,
            use_predictions=config.get("use_predictions", False),
            entailment_labels=entailment_labels,
            **cls._extra_kwargs_from_config(config),
        )

    @classmethod
    def _extra_kwargs_from_config(cls, config: dict) -> dict:
        """Override to supply benchmark-specific ``__init__`` kwargs."""
        return {}

    def _build_metric(self) -> Metric:
        """Return the concrete metric instance for this benchmark."""
        raise NotImplementedError

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        max_eval_n: Optional[int] = 1000,
        eval_seed: int = 42,
        cache_dir: Optional[str] = None,
        use_cached_expl: bool = False,
        use_hf_expl: bool = False,
    ):
        """Evaluate the benchmark using a given explanation method.

        Parameters mirror :meth:`Benchmark.evaluate`.
        """
        precomputed = self._resolve_precomputed_explanations(
            cache_dir=cache_dir,
            use_cached_expl=use_cached_expl,
            use_hf_expl=use_hf_expl,
        )
        explainer = (
            None
            if precomputed is not None
            else self._prepare_explainer(
                dataset=self.train_dataset,
                explainer_cls=explainer_cls,
                expl_kwargs=expl_kwargs,
            )
        )

        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=self._build_metric(),
            batch_size=batch_size,
            max_eval_n=max_eval_n,
            eval_seed=eval_seed,
            precomputed_explanations=precomputed,
        )
