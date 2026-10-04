#!/bin/bash
# Run from the root of the cloned LogiCo repo.
# Adds --accum_steps to train_logico.py so batch 14 fits in 24 GB as 2 x 7.
set -e
git apply "$(dirname "$0")/accum.patch"
grep -c accum_steps train_logico.py
