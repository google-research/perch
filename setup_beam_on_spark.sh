#!/bin/bash
# Copyright 2023 The BIRB Authors.
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
set -u

WORK_DIR="/tmp/beam"
SPARK_VERSION="3.4.0"
SPARK_NAME="spark-$SPARK_VERSION"
SPARK_ROOT="$SPARK_NAME-bin-hadoop3"
SPARK_BINARY="$SPARK_ROOT.tgz"
SPARK_DOWNLOAD_URL="http://archive.apache.org/dist/spark/$SPARK_NAME/$SPARK_BINARY"

SPARK_HOST=`hostname`
SPARK_PORT=7077
SPARK_LEADER_URL=spark://$SPARK_HOST:$SPARK_PORT
SPARK_CONF=spark.conf
SPARK_SECRET=`openssl rand -hex 20`

function setup_spark() {
  echo "Using work directory $WORK_DIR"
  mkdir -p $WORK_DIR
  if [ ! -d $WORK_DIR/$SPARK_ROOT ]; then
    echo "SETUP SPARK at $WORK_DIR/$SPARK_ROOT"
    pushd $WORK_DIR && curl $SPARK_DOWNLOAD_URL -o $WORK_DIR/$SPARK_BINARY  && tar -xvf $SPARK_BINARY
    if [ $? != 0 ]; then
      echo "ERROR: Unable to download Spark from $SPARK_DOWNLOAD_URL." \
            "Please make sure you have working internet and you have" \
            "curl(https://en.wikipedia.org/wiki/CURL) on your machine." \
            "Alternatively, you can also manually download Spark archive"\
            "and place it at $SPARK_DOWNLOAD_URL and extract Spark"\
            "to $WORK_DIR/$SPARK_ROOT"
      exit 1
    fi
    echo "SPARK SETUP DONE at $WORK_DIR/$SPARK_ROOT"
    popd
  fi
}

function start_spark() {
  echo "Starting Spark at $WORK_DIR/$SPARK_ROOT"
  pushd $WORK_DIR/$SPARK_ROOT
  ./sbin/stop-all.sh
  # Authenticate to avoid CVE-2018-17190
  # https://spark.apache.org/security.html
  echo -e "spark.authenticate true" > $SPARK_CONF
  echo -e "spark.authenticate.secret $SPARK_SECRET" > $SPARK_CONF
  echo -e "spark.master.rest.enabled true" > $SPARK_CONF
  # default web UI port (8080) is also used by Airflow
  ./sbin/start-master.sh -h $SPARK_HOST -p $SPARK_PORT --webui-port 8081 --properties-file $SPARK_CONF
  ./sbin/start-worker.sh $SPARK_LEADER_URL --properties-file $SPARK_CONF
  echo "Spark running from $WORK_DIR/$SPARK_ROOT"
  popd
}

function prepare_soundscapes() {
  for data_region in ssw hawaii coffee_farms high_sierras sierras_kahl peru; do
    poetry run tfds build -i birb.data.bird_taxonomy,birb.data.soundscapes \
      --datasets=soundscapes \
      --config=${data_region}_full_length \
      --beam_pipeline_options="runner=SparkRunner,spark_rest_url=http://localhost:6066,environment_type=LOOPBACK"
    done 
}

function stop_spark() {
  echo "Stopping Spark at $WORK_DIR/$SPARK_ROOT"
  pushd $WORK_DIR/$SPARK_ROOT
  ./sbin/stop-all.sh
  popd
}

function main(){
  setup_spark
  start_spark
  prepare_soundscapes
  stop_spark
}

main $@
