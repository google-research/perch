# Bird Separation+Classification Models

The code in this directory is for using the separation and classification models
discussed in our [paper](https://arxiv.org/abs/2110.03209).

Much of the code here is for running inference or embedding large datasets,
so we provide some example usage below to get you started.

You can download the model files for either `lorikeet` (covering 87 output
classes, including common birds from East Australia and parrot species
worldwide) or `sierras` (covering 89 classes appearing in the California Sierra
Nevadas).  The models can be downloaded with `gsutil` like so:

```
gsutil -m cp -r \
  "gs://chirp-public-bucket/birbsep_paper" .
```

The bird separation models for 4 or 8 output channels can be downloaded with:

```
gsutil -m cp -r \
  "gs://gresearch/sound_separation/bird_mixit_model_checkpoints" .
```

## Simple Usage for Colab/Jupyter Notebooks

To load and run the models simply adapt the following snippets.
`$CLASSIFIER_PATH` should point to the path containing the classifiers. In
particular, it should contain subdirectories named `run_00` through `run_04`.

Here's how to load a single classifier and use it to embed some audio:

```
import numpy as np
import tensorflow
from chirp.birb_sep_paper import model_utils

tf = tensorflow.compat.v1

model_path = '$CLASSIFIER_PATH/run_00'
classy = model_utils.load_classifier_state(model_path)

fake_audio = np.zeros([1, 5*22050])
# For the `lorikeet` model, there are 87 output classes, so the hints should
# have shape [Batch, 87]. The Sierras model has 89 output species.
hints = np.ones([1, 87])

# Now call the model and get embeddings.
embeddings = model_utils.model_embed(
    fake_audio, classy, hints, 'hidden_embedding')
```

Easy!

To classify audio using the single model we just loaded:

```
key = 'label'  # one of ['label', 'genus', 'family', 'order', 'is_bird']
melspec, logits = model_utils.ensemble_classify(
    fake_audio, {'run_00': classy}, hints, key)
```

To classify audio using the full ensemble of five trained models:
```
ensemble = model_utils.load_classifier_ensemble('$MODEL_PATH')
melspec, logits = model_utils.ensemble_classify(
  fake_audio, ensemble, hints, 'label')
reduced_logits = np.mean(logits, axis=1)
```

The `melspec` will have shape `[batch, 501, 160]` and contains the
PCEN'ed melspectrograms exactly as fed to the classifier. These can be helpful
for judging whether a vocalization was 'visible' or not.

The logits will have shape `[batch, num_classifiers, num_species]`, and gives
the logits for each classifier in the ensemble. Averaging the ensemble logits
will give stronger results than using an individual model's logits.


## Separation

Loading and running a separation model is similar. However the functions
provided are set up to run on a single long audio clip.

```
long_audio = np.zeros([30 * 22050])
separator = model_utils.load_separation_model($SEPARATION_MODEL_PATH)
sep_chunks, raw_chunks = model_utils.separate_windowed(
  long_audio, separator, hop_size=5.0, window_size=5)
```

The `sep_chunks` will have shape `[6, 4, 5*22050]`. The function divides the
input into 5-second chunks, then runs the separator, which produces four
separated output channels. The `raw chunks` have shape `[6, 1, 5 * 22050]` and
contain the original audio, windowed to match the separated audio.

To both separate and classify, use the `separate_classify` function with the
loaded separator and ensemble:

```
melspecs, logits = separate_classify(long_audio, ensemble, separator, hints,
    hop_size=5.0)
```

This uses the method in the paper: The raw audio is stacked on the separated
channels and all channels are classified. We take the max logit over the
channels, and then the mean logit over the ensemble. Then the result should have
shape `[6, 87]`.
