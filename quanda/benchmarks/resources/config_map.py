"""Here benchmark handles will be match to the config files."""

from importlib.resources import files

_unit = "tests/assets/unit_bench_cfgs"
_prefix = "3d2d93d-default"

config_map: dict = {
    "mnist_linear_datamodeling_unit": (f"{_unit}/{_prefix}_LDS.yaml"),
    "mnist_top_k_cardinality_unit": (f"{_unit}/{_prefix}_ClassDetection.yaml"),
    "mnist_subclass_detection_unit": (
        f"{_unit}/{_prefix}_SubclassDetection.yaml"
    ),
    "mnist_mixed_datasets_unit": (f"{_unit}/{_prefix}_MixedDatasets.yaml"),
    "mnist_mislabeling_detection_unit": (
        f"{_unit}/{_prefix}_MislabelingDetection.yaml"
    ),
    "mnist_shortcut_detection_unit": (
        f"{_unit}/{_prefix}_ShortcutDetection.yaml"
    ),
    "mnist_class_detection_unit": (f"{_unit}/{_prefix}_ClassDetection.yaml"),
    "mnist_model_randomization_unit": (
        f"{_unit}/{_prefix}_ClassDetection.yaml"
    ),
    "mnist_top_k_cardinality": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("ad1b983-default_ClassDetection.yaml"),
    "mnist_subclass_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("ad1b983-default_SubclassDetection.yaml"),
    "mnist_mixed_datasets": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("ad1b983-default_MixedDatasets.yaml"),
    "mnist_mislabeling_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("124627f-default_MislabelingDetection.yaml"),
    "mnist_shortcut_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("124627f-default_ShortcutDetection.yaml"),
    "mnist_class_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("ad1b983-default_ClassDetection.yaml"),
    "mnist_model_randomization": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("ad1b983-default_ClassDetection.yaml"),
    "mnist_linear_datamodeling": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("0b3b2bb-default_LDS.yaml"),
    "cifar_class_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("bdb919e-default_ClassDetection.yaml"),
    "cifar_top_k_cardinality": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("bdb919e-default_ClassDetection.yaml"),
    "cifar_model_randomization": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("bdb919e-default_ClassDetection.yaml"),
    "cifar_subclass_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("bdb919e-default_SubclassDetection.yaml"),
    "cifar_mixed_datasets": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("bdb919e-default_MixedDatasets.yaml"),
    "cifar_shortcut_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("bdb919e-default_ShortcutDetection.yaml"),
    "cifar_mislabeling_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("bdb919e-default_MislabelingDetection.yaml"),
    "cifar_linear_datamodeling": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("a701608-default_LDS.yaml"),
    "qnli_class_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("e59b08c-bert_qnli_ClassDetection.yaml"),
    "qnli_top_k_cardinality": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("e59b08c-bert_qnli_ClassDetection.yaml"),
    "qnli_model_randomization": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("e59b08c-bert_qnli_ClassDetection.yaml"),
    "qnli_mislabeling_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("e59b08c-bert_qnli_MislabelingDetection.yaml"),
    "qnli_linear_datamodeling": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("e59b08c-bert_qnli_LDS.yaml"),
    "qnli_mixed_datasets": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("99a4f7b-bert_qnli_MixedDatasets.yaml"),
    "nanogpt_trex_mrr": files("quanda.benchmarks.resources.configs").joinpath(
        "nanogpt_trex_MRR.yaml"
    ),
    "nanogpt_trex_recall_at_k": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("nanogpt_trex_RecallAtK.yaml"),
    "nanogpt_trex_tail_patch": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("nanogpt_trex_TailPatch.yaml"),
}
