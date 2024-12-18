# BIRB Evaluation

This folder contains the evaluation framework for the BIRB benchmark retrieval task.
(This is NOT the script to generate classification metrics.)

For details on setting up the benchmark and evaluation protocol, please refer to this [brief readme](https://docs.google.com/document/d/1RasVkxIKKlUToFlJ8gZxaHqIE-mMy9G1MZwfK98Gb-I) with instructions.

To build the BIRB evaluation data, after [installing](../../README.md#installation) the `chirp` package, run the following command from the repository's root directory:

```bash
poetry run tfds build -i chirp.data.bird_taxonomy,chirp.data.soundscapes \
    soundscapes/{ssw,hawaii,coffee_farms,sierras_kahl,high_sierras,peru}_full_length \
    bird_taxonomy/{downstream_full_length,class_representatives_slice_peaked}
```

The process should take 36 to 48 hours to complete and use around 256 GiB of disk space.

## Running eval.py

The evaluation script requires a config file that specifies the model, datasets, and evaluation parameters.

Run the script using:

    poetry run python birb/eval/eval.py -- --config=<PATH_TO_EVAL_CONFIG_FILE>


### Required Config Settings

[Read what your config must specify.](https://docs.google.com/document/d/1RasVkxIKKlUToFlJ8gZxaHqIE-mMy9G1MZwfK98Gb-I)

### Outputs

After executing the evaluation pipeline, output evaluation metrics for each
species results are written:

* `eval_species`: Species identifier
* `average_precision`: AP score
* `roc_auc`: ROC AUC score
* `num_pos_match`: Number of positive matches
* `num_neg_match`: Number of negative matches
* `eval_set_name`: Name of evaluation dataset

Results are written to the path specified by `config.write_results_dir`,
in CSV format.