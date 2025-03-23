#!/usr/bin/env bash

# run the configure_and_build script using the apptainer environment
# (bind the /home/shared filesystem so it is visible inside the container)
apptainer run --bind /home/shared:/home/shared \
  --bind /apps:/apps \
  --bind /scratch-local:/scratch-local \
  build_environment/build_environment.sif \
  ./scripts/configure_build_run.sh
