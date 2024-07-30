#!/bin/bash

python dataset.py # build train dataset

python main.py --model GLD_finite # train model
