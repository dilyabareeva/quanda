# Preparing Benchmarks for Quanda
Here, we go through the steps for creating a benchmark for quanda.

Currently the scripts rely totally on lightning for training and HuggingFace for datasets.

Overall these are the steps:

1. Make necessarry additions to the quanda codebase for the benchmarks.
2. Make necessary additions to the scripts.
3. Train models.
4. Create benchmark files.
5. Update benchmark URLs in quanda.

Below we go through each one individually:

## Step 1: Additions to the codebase

- <details><summary>Decide on a keyword as a name for your Lightning module:</summary> This module will include all details about the model architecture and the training process. It is also possible to use one of the already existing modules, but the current modules are named after the datasets they were designed for. In the rest of the document, this keyword will be referred to as `module_name` </details>
- <details><summary></summary></details>

## Step 2: Additions to the scripts


- <details><summary>Decide on a dataset name tag:</summary>

  This is a keyword which will be used internally and as the name of the directory for HF caching. In the rest of this documents, this keyword will be referred to as `ds_name`.</details>

- <details><summary>Populate the `datasets_metadata` dictionary with the required hyperparameters for your benchmarks:</summary>
  These are currently:
  * `hf_tag`: The name used to download the dataset from HuggingFace.
  * `validation_size`: Number of datapoints from test split to use as validation set.
  * `test_split_name`: The name of the test split for huggingface. Probably "test" or "val".
  * `num_classes`: Number of classes in the dataset.
  * `shortcut_cls`: The class to use as shortcut class in `ShorcutDetection` benchmark.
  * `shortcut_probability`: Probability of shortcutting to use with `ShortcutDetection` benchmark.
  * `num_groups`: The number of superclasses to use in `SublabelDetection` benchmark.
  * `mislabeling_probability`: Probabilty of mislabeling to use with `MislabelingDetection` benchmark.
  * `adversarial_cls`: The class to assign adversarial images to, while using `MixedDatasets` benchmark.
  * `adversarial_dir_url`: URL to the adversarial image directory with the correct filename.

  The final 6 items are optional and can be set to None if the corresponding benchmarks are not of interest.
  </details>
- <details><summary>Add `module_name` to the `module_kwargs` dictionary in the `load_pl_module` function in train_model.py: </summary> 
  The value associated to the key `module_name` should be a dictionary containing the training hyperparameters that you want to control during the training phase. This dictionary will be used to initialize the Lightning module that was coded in step 1. If you are adding new hyperparameters to be passed here, you should get the value using argparse and change the script accordingly.</details>