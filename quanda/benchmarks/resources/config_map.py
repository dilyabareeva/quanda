"""Here benchmark handles will be match to the config files."""

from importlib.resources import files

config_map: dict = {
    "mnist_top_k_cardinality_unit": "tests/assets/unit_bench_cfgs/20fba38-default_ClassDetection.yaml",
    "mnist_subclass_detection_unit": "tests/assets/unit_bench_cfgs/20fba38-default_SubclassDetection.yaml",
    "mnist_mixed_datasets_unit": "tests/assets/unit_bench_cfgs/20fba38-default_MixedDatasets.yaml",
    "mnist_mislabeling_detection_unit": "tests/assets/unit_bench_cfgs/20fba38-default_MislabelingDetection.yaml",
    "mnist_shortcut_detection_unit": "tests/assets/unit_bench_cfgs/20fba38-default_ShortcutDetection.yaml",
    "mnist_class_detection_unit": "tests/assets/unit_bench_cfgs/20fba38-default_ClassDetection.yaml",
    "mnist_model_randomization_unit": "tests/assets/unit_bench_cfgs/20fba38-default_ClassDetection.yaml",
    "mnist_top_k_cardinality": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("b6912b6-default_ClassDetection.yaml"),
    "mnist_subclass_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("88c491d-default_SubclassDetection.yaml"),
    "mnist_mixed_datasets": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("b6912b6-default_MixedDatasets.yaml"),
    "mnist_mislabeling_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("b6912b6-default_MislabelingDetection.yaml"),
    "mnist_shortcut_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("88c491d-default_ShortcutDetection.yaml"),
    "mnist_class_detection": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("b6912b6-default_ClassDetection.yaml"),
    "mnist_model_randomization": files(
        "quanda.benchmarks.resources.configs"
    ).joinpath("b6912b6-default_ClassDetection.yaml"),
}
