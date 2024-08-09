#!/bin/bash

# Navigate to project directory
# cd /Users/anseunghwan/Documents/GitHub/CryptoVAE-online
cd /Users/an-seunghwan/Documents/GitHub/CryptoVAE-online

# build train dataset
# /opt/anaconda3/envs/crypto/bin/python dataset.py 
/opt/anaconda3/envs/deep/bin/python dataset.py 

# train model
# /opt/anaconda3/envs/crypto/bin/python main.py --model GLD_finite 
/opt/anaconda3/envs/deep/bin/python main.py --model GLD_finite 

# Git commit and push
git add .
current_date=$(date +"%Y-%m-%d") # Get today's date
git commit -m "Automated update on ${current_date}"
git push