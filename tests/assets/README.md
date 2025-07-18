# Training a new unit test suite

- Run the script `scripts/mnist_unit_tests.sh` to generate a new test suite.
- The script will create new configuration files in `tests/assets/mnist_test_suite_3`, or in a different directory if specified in the training script.
- Replace the paths to configuration files in `quanda/benchmarks/resources/config_map.py` with the new paths. As a rule, is enough to just replace the commit prefix of the files.
- Git add the new files and commit them to the repository.