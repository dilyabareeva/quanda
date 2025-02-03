# Preparing Benchmarks for Quanda
Here, we go through the steps for creating a benchmark for quanda.

Currently the scripts rely totally on lightning for training and HuggingFace for datasets.

Overall these are the steps:

1. Make necessarry additions to the quanda codebase for the benchmarks.
2. Make necessary additions to the scripts.
3. Train models.
4. Create benchmark files.
5. Test that benchmarks initialize.
6. Update benchmark URLs in quanda.

Below we go through each one individually:

## Step 1: Additions to the codebase

- <details><summary>Decide on a keyword as a name for your Lightning module:</summary>

   This module will include all details about the model architecture and the training process. It is also possible to use one of the already existing modules, but the current modules are named after the datasets they were designed for. In the rest of the document, this keyword will be referred to as `module_name` </details>
- <details><summary>Implement Lightning module:</summary>


  Implement the training details and model architecture in a Lightning Module in benchmarks/resources/modules.py
</details>

- Add `module_name` and the class implementation to the `pl_modules` dictionary in benchmarks/resources/modules.py

- <details><summary>Decide on a dataset name tag:</summary>

  This is a keyword which will be used internally and as the name of the directory for HF caching. In the rest of this documents, this keyword will be referred to as `ds_name`.</details>

- <details><summary>

  Add the following keys `sample_transforms` dictionary in benchmarks/resources/sample_transforms.py with the required dataset transforms for your benchmarks :</summary>

  The keys all start with `ds_name` followed by the suffixes listed below:
  * `_transforms`: The standard resize/normalize transformations for image datasets. Simplest dataset to use directly with the model.
  * `_adversarial_transform`: The transform to use for adversarial images when using `MixedDataset` benchmark.
  * `_shortcut_transform`: The shortcut transform to use for the `ShortcutDetection` dataset. This will get the raw dataset items and should output the same type of object, which will then be passed to the regular transform for the dataset.

  The final 2 items are optional and can be set to None if the corresponding benchmarks are not of interest.
  </details>


## Step 2: Additions to the scripts




- <details><summary>

  Populate the `datasets_metadata` dictionary in scripts/train_model.py with the required hyperparameters for your benchmarks with the `ds_name` key:</summary>

  The value for the `ds_name` key is a dictionary. The required keys are currently:
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
- <details><summary>

  Add `module_name` to the `module_kwargs` dictionary in the `load_pl_module` function in train_model.py: </summary>


  The value associated to the key `module_name` should be a dictionary containing the training hyperparameters that you want to control during the training phase. This dictionary will be used to initialize the Lightning module that was coded in step 1. If you are adding new hyperparameters to be passed here, you should get the value using argparse and change the script accordingly.</details>

- <details><summary>(Optional) Add additional checks:</summary>

  After training, some checks are run on the models depending on the type of dataset the model was trained on. You can add functions in model_sanity_checks.py and populate the `sanity_checks` and `func_params` in the `run_model_sanity_checks` function. Depending on the complexity of your checks, you may need to make additions to the scripts to pass relevant information to the `run_model_sanity_checks` function. </details>. If you don't want to run any checks, you can ignore this step. You can run the script with the --ignore_sanity_checks option to skip this step altogether.

## Step 3: Training

While training, we can change some hyperparameters while others are fixed in the current design. Currently, schedulers and optimizers are defined inside the LightningModule implementation. But learning rates, number of epochs, using pretrained models and augmentations are handled. Schedulers and optimizers can be handled similarly through parameter passing while initializing the LightningModules in the `load_pl_module` function.

Augmentations are controlled through a string of predefined augmentation keys, seperated with underscores (_). Thus setting `--augmentation flip_crop_rotate` fill concatenate 3 augmentations as defined in the `load_augmentation` function. Currently only image augmentations are included. Augmentations can be avoided by not giving an `--augmentation` at all (using `augmentation=None`).

The script requires you to decide on some paths for caching:
- `dataset_cache_dir`: Path to use for caching HF dataset.
- `metadata_root`: A path to cache details relating to the benchmarks (i.e. validation/test indices, mislabeled indices, mislabeling labels etc.) This should be static and populated with job outputs as you train more models with different datasets. Then, when you run the job again, the same datasplits and data manipulations will be done. This folder will also include files needed to create the benchmark so will be needed in step 4.
- `output_path`: Directory to save the job outputs. If not given, `metadata_root` is used. Otherwise, you should copy the outputs to the `metadata_root`.

Different kinds of datasets are associated with keywords. The `dataset_type` input takes these keywords as values: vanilla, subclass, mislabeled, mixed.

Run the train_model.py script to train models:

```
python scripts/train_model.py
        --dataset_name <ds_name>
        --dataset_type vanilla | mislabeled | subclass | mixed | shortcut
        --dataset_cache_dir, <ds_cache_path>,
        --augmentation <augmentation_string>
        --weight_decay 0.0
        --metadata_root <metadata_root>
        --output_path <output_path>
        --device cpu
        --module_name <module_name>
        --pretrained
        --epochs 50
        --lr 0.01
        --batch_size 64
        --save_each 2 # defaults to the value of validate_each
        --validate_each 1
```

## Step 4: Creating the benchmark file
After training the models, put them in seperate directories. We also populate the `metadata_path` with the required indices. We can now use the make_benchmark.py script to create the benchmark files.

The `benchmark_name` input takes values as in "subclass_detection", "shortcut_detection", "topk_overlap" etc., and the options are listed in the `benchmark_urls` array.

```
python scripts/make_benchmark.py
        --benchmark_name <benchmark_name>
        --dataset_name <ds_name>,
        --dataset_cache_dir <ds_cache_dir>,
        --dataset_type <ds_type>
        --metadata_root <metadata_root>
        --output_path <output_path>
        --seed <seed>
        --device cpu
        --module_name <module_name>
        --checkpoints_dir <checkpoints_path>
```

Step 5: Initialize benchmarks

After the benchmark dictionaries are created, save them in a directory. Then, test_benchmarks.py can be used to make sure benchmarks initialize without problems from those benchmark dictionaries.

Step 6: Upload and publicize

Alright, we can host the benchmarks somewhere and update the urls in the quanda codebase.
