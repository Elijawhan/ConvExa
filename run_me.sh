#!/usr/bin/env bash

# run the configure_and_build script using the apptainer environment
# (bind the /home/shared filesystem so it is visible inside the container)
# apptainer run --bind /home/shared:/home/shared \
#   --bind /apps:/apps \
#   --bind /scratch-local:/scratch-local \
#   build_environment/build_environment.sif \
rm -rf *.qsub_out
qsub ./scripts/configure_build_run.sh


# wait for the job to get picked up and start producing output
until [ -f *.qsub_out ]
do
  sleep 1
done

# open the output file and follow the file as new output is added
less +F *.qsub_out