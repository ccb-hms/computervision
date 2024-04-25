#!/bin/bash
# --tunnel 7488:7488
echo "Requesting interactive CPU partition and 16G memory"
srun -p interactive -c 1 -t 0-6:00 --pty --mem 16G /bin/bash
