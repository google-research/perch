# Chirp Configurations

## The Lifecycle of Configuration Objects

We use ConfigDict objects for configuration. For the most part, the
configuration mirrors keyword arguments for functions called during program
execution.

Occasionally there are keyword arguments with scope greater than a single
function; a good example is `sample_rate_hz`, which is an argument to many
functions, and needs to be consistent across all function calls.

Note that sometimes config values are changed after the configuration is initially created. Examples are test configs (where we might want to reduce model or input size to keep tests quick) or when performing hyperparameter sweeps.

To facilitate this, we can use FieldReferences, which allow values in the
ConfigDict to change automatically when the referred value is changed.
The best way to use references is by creating a reference to the target field
using `get_ref`:

```
cfg = config_dict.ConfigDict()
cfg.window_size_s = 5
cfg.sample_rate_hz = 16000

cfg.input_size = cfg.get_ref('window_size_s') * cfg.sample_rate_hz
print(cfg.input_size)  # 80000

# Now if we change the window_size_s, the input size will change automagically.
cfg.window_size_s = 1
print(cfg.input_size)  # 16000

# But if we change the sample_rate_hz, the input_size stays the same, because
# we did not use a reference.
cfg.sample_rate_hz = 11050
print(cfg.input_size)  # 16000
```

Now, FieldReferences are not numbers, and will cause downstream applications
to panic if they expect an integer and get a FieldReference. The
`config_utils.parse_config` will automatically resolve all FieldReferences
to their final values.
