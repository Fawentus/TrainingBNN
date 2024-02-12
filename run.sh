#!/bin/bash

echo "Start training backpropagation"
python3 ./src/backpropagation/main.py > ./data/models/backpropagation/README3.md

echo "Start training evolutionary"
python3 ./src/evolutionary/main.py > ./data/models/evolutionary/README3.md

echo "Start training genetic"
python3 ./src/genetic/main.py > ./data/models/genetic/README3.md

echo "Start training ordinary"
python3 ./src/ordinary/main.py > ./data/models/ordinary/README3.md
