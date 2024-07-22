#!/bin/bash

# This bash script activates the pipenv environment
# in the parent directory.

# Activate python version 3.10.11
module load python/3.10.11
# Load gcc and cuda library
module load gcc/9.2.0 && module load cuda/12.1
# Set the data directory for the computer vision notebooks
# Edit this variable if you want to have your data in a different directory.
export DATA_ROOT="$HOME/data"
mkdir -p $DATA_ROOT
# Activate the environment
pipenv shell
