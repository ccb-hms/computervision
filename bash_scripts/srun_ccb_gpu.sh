#!/bin/bash
# --tunnel 7488:7488
echo "Requesting interactive CCB GPU partition."
srun -p gpu_ccb --account=gentleman_rcg7_contrib -c 16 -t 1-00:00 --pty --mem 128G --gres=gpu:1 /bin/bash
