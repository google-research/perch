# coding=utf-8
# Copyright 2022 The Chirp Authors.
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

You can also pass a `ConfigDict` as an argument:

  foo_config = ConfigDict()
  foo_config.bar = 4
  config.foo = callable_config("my_module.Foo", config.foo_config)

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
from typing import Any, Dict
from ml_collections import config_dict

_CALLABLE = "__constructor"
_KWARGS = "__config"
_OBJECT = "__object"


def callable_config(callable_: str, *args: config_dict.ConfigDict,
                    **kwargs: Any) -> config_dict.ConfigDict:
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


def parse_config(config: config_dict.ConfigDict,
                 globals_: Dict[str, Any]) -> config_dict.ConfigDict:
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
    if set(value.keys()) == {_CALLABLE, _KWARGS}:
      return eval(value[_CALLABLE], globals_)(  # pylint: disable=eval-used
          **parse_config(value[_KWARGS], globals_).to_dict())
    elif set(value.keys()) == {_OBJECT}:
      return eval(value[_OBJECT], globals_)  # pylint: disable=eval-used
    else:
      return parse_config(value, globals_)

  with config.ignore_type():
    for key, value in config.items():
      if isinstance(value, config_dict.ConfigDict):
        config[key] = _parse_value(value)
      elif isinstance(value, (list, tuple)):
        config[key] = [
            _parse_value(v) if isinstance(v, config_dict.ConfigDict) else v
            for v in value
        ]
    return config
