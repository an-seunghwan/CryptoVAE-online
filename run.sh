#!/bin/bash

exp=(
"--model GLD_finite"
)

for i2 in ${!exp[*]}; do
    python main.py ${exp[$i2]}
done
