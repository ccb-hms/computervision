#!/bin/bash
# --tunnel 7488:7488
echo "Requesting interactive GPU partition and 16G memory"
srun -p gpu -c 1 -t 0-6:00 --pty --mem 16G --gres=gpu:1 /bin/bash
