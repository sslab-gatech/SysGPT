#!/bin/bash

temp=(0.0 0.1 0.2 0.3 0.4 0.5 0.7)
trial=(1 3 5 10)

mkdir -p ./auto_log

for a1 in "${temp[@]}"; do
  for a2 in "${trial[@]}"; do
    echo "Running: python3 -u eval.py $a1 $a2 | tee ./auto_log/temp$a1-best$a2.log"
    python3 -u eval.py "$a1" "$a2" | tee ./auto_log/temp$a1-best$a2.log
  done
done
