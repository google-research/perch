#!/bin/bash

blaze run -c opt --copt=-mavx \
  //third_party/py/chirp/data/soundscapes:download_and_prepare -- \
  --gfs_user=kakapo --datasets=soundscapes --flume_exec_mode=IN_PROCESS
