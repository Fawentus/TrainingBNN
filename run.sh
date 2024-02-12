#!/bin/bash

echo "Start training backpropagation"
python ./src/backpropagation/main.py > ./data/models/backpropagation/README3.md

echo "Start training evolutionary"
python ./src/evolutionary/main.py > ./data/models/evolutionary/README3.md

echo "Start training genetic"
python ./src/genetic/main.py > ./data/models/genetic/README3.md

echo "Start training ordinary"
python ./src/ordinary/main.py > ./data/models/ordinary/README3.md
