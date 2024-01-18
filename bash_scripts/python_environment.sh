#!/bin/bash
module load gcc/9.2.0
module load cuda/11.7
module load python/3.10.11

repo_dir="/home/aaw10/gitrepos/bigsmile"
cd ${repo_dir}
pipenv shell
