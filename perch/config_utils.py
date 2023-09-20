# coding=utf-8
# Copyright 2023 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to be able to construct Python objects from configurations.

First use `callable_config` to construct a `ConfigDict` as follows:

  config.foo = callable_config("my_module.Foo", bar=4)

This will construct a `ConfigDict` that looks as follows:

  ConfigDict({
    'foo': ConfigDict({
      "__constructor": "my_module.Foo",
      "__config": ConfigDict({
        "bar": 4
      })
    })
  })

This configuration dictionary can be serialized and the keys can even be
overridden on the command line. The objects can be constructed by calling
`parse_config(config, {"my_module": my_module})` which returns the following:

  ConfigDict({
    'foo': my_module.Foo(bar=4)
  })
"""
from typing import Any
from ml_collections import config_dict

_CALLABLE = "__constructor"
_KWARGS = "__config"
_OBJECT = "__object"


def callable_config(
    callable_: str, *args: config_dict.ConfigDict, **kwargs: Any
) -> config_dict.ConfigDict:
  """Create a configuration for constructing a Python object.

  Args:
    callable_: A string that resolves to a Python object to call in order to
      construct the configuration value.
    *args: Configuration dictionaries containing keyword arguments to pass to
      the callable.
    **kwargs: The keyword arguments to pass to the callable.

  Returns:
    A ConfigDict object containing the callable and its arguments. This
    dictionary can later be parsed by the `parse_config` function.
  """
  kwargs = config_dict.ConfigDict(kwargs)
  for arg in args:
    kwargs.update(arg)
  return config_dict.ConfigDict({_CALLABLE: callable_, _KWARGS: kwargs})


def object_config(object_: str) -> config_dict.ConfigDict:
  """Create a configuration for a Python object.

  Args:
    object_: A string that resolve to a Python object.

  Returns:
    A ConfigDict object containing the name of the object. This dictionary can
    later be parsed by the `parse_config` function.
  """
  return config_dict.ConfigDict({_OBJECT: object_})


def either(object_a: Any, object_b: Any, return_a: bool) -> Any | None:
  """Returns returns object_a if `predicate` is True and object_b otherwise.

  While trivial in appearance, this function can be used in conjunction with
  `callable_config` to implement control flow with a boolean
  `config_dict.FieldReference`:

  config.some_attribute = callable_config(
      'either',
      object_a=callable_config(...),
      object_b=callable_config(...),
      return_a=config.get_ref('some_boolean_field_reference')
  )

  Args:
    object_a: The first object to (maybe) return.
    object_b: The second object to (maybe) return.
    return_a: Whether to return object_a (True) or object_b (False).

  Returns:
    object_a or object_b, depending on `return_a`.
  """
  return object_a if return_a else object_b


def get_melspec_defaults(config: config_dict.ConfigDict) -> tuple[Any, Any]:
  """Determines the default melspectrogram kernel size and nftt values.

  Args:
    config: The base ConfigDict. Expected to contain 'sample_rate_hz' and
      'frame_rate_hz' attributes.

  Returns:
    The default kernel size and nftt values.

  Raises:
    ValueError, if the default kernel size is determined to be larger than 4096.
      If this is the case, the config is expected to define a default nftt value
      directly.
  """
  melspec_stride = config.get_ref("sample_rate_hz") // config.get_ref(
      "frame_rate_hz"
  )
  # This gives 50% overlap, which is optimal for the Hanning window.
  # See Heinz, et al: "Spectrum and spectral density estimation by the
  # Discrete Fourier transform (DFT), including a comprehensive list of window
  # functions and some new flat-top windows", Section 10.
  # https://holometer.fnal.gov/GH_FFT.pdf
  # In brief, 50% overlap gives no amplitude distortion, and minimizes the
  # overlap correlation. Longer windows average over longer time periods,
  # losing signal locality, and are also more expensive to compute.
  melspec_kernel_size = 2 * melspec_stride
  # nfft is preferably the smallest power of two containing the kernel.
  # This yields a no-nonsense FFT, implemented everywhere.
  # Note that we can"t use fancy math like ceil(log2(ks)) on field references...
  if melspec_kernel_size <= 256:
    melspec_nfft = 256
  elif 256 < melspec_kernel_size <= 512:
    melspec_nfft = 512
  elif 512 < melspec_kernel_size <= 1024:
    melspec_nfft = 1024
  elif 1024 < melspec_kernel_size <= 2048:
    melspec_nfft = 2048
  elif 2048 < melspec_kernel_size <= 4096:
    melspec_nfft = 4096
  else:
    raise ValueError("Large kernel {kernel_size}; please define nfft.")

  return melspec_kernel_size, melspec_nfft


def parse_config(
    config: config_dict.ConfigDict, globals_: dict[str, Any]
) -> config_dict.ConfigDict:
  """Parse a configuration.

  This handles nested configurations, as long as the values are callables
  created using `callable_config`, or if the values are lists or tuples
  containing elements created the same way.

  Args:
    config: A configuration object, potentially containing callables which were
      created using `callable_config`.
    globals_: The dictionary of globals to use to resolve the callable.

  Returns:
    The parsed configuration dictionary.
  """

  def _parse_value(value: config_dict.ConfigDict) -> Any:
    if isinstance(value, dict):
      value = config_dict.ConfigDict(value)
    if isinstance(value, config_dict.ConfigDict):
      if set(value.keys()) == {_CALLABLE, _KWARGS}:
        return _parse_value(
            eval(value[_CALLABLE], globals_)(  # pylint: disable=eval-used
                **parse_config(value[_KWARGS], globals_)
            )
        )
      elif set(value.keys()) == {_OBJECT}:
        return _parse_value(eval(value[_OBJECT], globals_))  # pylint: disable=eval-used
      else:
        return parse_config(value, globals_)
    elif isinstance(value, config_dict.FieldReference):
      return value.get()
    else:
      return value

  with config.ignore_type():
    for key, value in config.items():
      # We purposefully only attempt to parse values inside list and tuple
      # instances (and not e.g. namedtuple instances, since optax defines
      # GradientTransformation as a namedtuple and we don't want to parse its
      # values), which precludes using isinstance(value, (list, tuple)).
      if type(value) in (list, tuple):
        config[key] = type(value)(_parse_value(v) for v in value)
      else:
        config[key] = _parse_value(value)
    return config
