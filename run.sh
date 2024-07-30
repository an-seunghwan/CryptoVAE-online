#!/bin/bash

# Activate virtual environment
source /opt/anaconda3/envs/crypto/bin/activate

# Navigate to project directory
cd /Users/anseunghwan/Documents/GitHub/CryptoVAE-online

# build train dataset
python dataset.py 

# train model
python main.py --model GLD_finite 

# Git commit and push
git add .
current_date=$(date +"%Y-%m-%d") # Get today's date
git commit -m "Automated update on ${current_date}"
git push