#!/usr/bin/env bash

python3 -m generate_cpdags.py --densities .1 .2 .5 .9 --nnodes 3, 5, 10, 30, 50, 70, 90, 110 --ndags 2000