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

r"""XManager script for launching Kakapo training on Borg.

    gxm third_party/py/chirp/google/xm_launch.py \
      --xm_resource_alloc=group:brain/chirp -- \
      --config=third_party/py/chirp/configs/baseline.py
"""
import getpass
import os
from typing import Sequence

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from xmanager import xm
from xmanager import xm_abc
from xmanager.contrib.internal import artifacts
from xmanager.contrib.internal import tensorboard
from xmanager.contrib.internal import xm_jax
from google3.learning.deepmind.xmanager import hyper

_CONFIG = config_flags.DEFINE_config_file("config")
_LOGDIR = flags.DEFINE_string("logdir", "/cns/lu-d/home/{user}/{xid}",
                              "Logging and checkpointing directory.")
flags.mark_flags_as_required(["config"])


def get_hyper():

  def _prepend_config(config):
    return {f"config.{k}": v for k, v in config.items()}

  return map(
      _prepend_config,
      hyper.product([
          hyper.zipit([
              hyper.sweep("model_config.bandwidth", hyper.discrete([0, 40])),
              hyper.sweep("model_config.band_stride", hyper.discrete([0, 30])),
          ]),
          hyper.sweep("model_config.random_low_pass", hyper.boolean()),
          hyper.sweep("model_config.robust_normalization", hyper.boolean()),
          hyper.sweep("model_config.melspec_config.scaling",
                      hyper.categorical(["pcen", "log", "raw"])),
          hyper.sweep("rng_seed", hyper.discrete([0, 1, 2]))
      ]))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with xm_abc.create_experiment(
      experiment_title="Initial test for Kākāpō.") as experiment:

    # NOTE: py_flags because otherwise GPU jobs balk at TPU flags
    args = xm_jax.JaxFlags().py_flags()

    # Load config and get the filename
    config_filename = flags.FLAGS["config"].config_filename

    # Package config file
    config_resource = xm_abc.Fileset(files={config_filename: config_filename})
    args["config"] = config_resource.get_path(config_filename,
                                              xm_abc.Borg.Spec())

    # Pass on any config flags
    args.update({
        key: value
        for key, value in flags.FLAGS.flag_values_dict().items()
        if key.startswith("config.")
    })

    # Create logdir and pass on
    logdir = _LOGDIR.value.format(
        user=getpass.getuser(), xid=experiment.experiment_id)
    args["logdir"] = logdir
    artifacts.create_artifact(
        experiment.experiment_id,
        artifact_type=artifacts.Type.ARTIFACT_TYPE_DIRECTORY,
        artifact=logdir,
        description="logdir")

    [executable] = experiment.package([
        xm.bazel_binary(
            label="//third_party/py/chirp:main",
            dependencies=[config_resource],
            bazel_args=xm_abc.bazel_args.gpu(),
            args=args,
            executor_spec=xm_abc.Borg.Spec())
    ])
    executor = xm_abc.Borg(
        requirements=xm.JobRequirements(
            V100=1,
            service_tier=xm.ServiceTier.PROD,
        ))

    tensorboard.add_tensorboard_corp(experiment, logdir)

    async def make_job(work_unit: xm.WorkUnit, **args):
      args["workdir"] = os.path.join(logdir, str(work_unit.work_unit_id))
      job = xm.Job(executable, executor, args=args)
      work_unit.add(job)

    for args in get_hyper():
      experiment.add(make_job, args=args)


if __name__ == "__main__":
  app.run(main)
