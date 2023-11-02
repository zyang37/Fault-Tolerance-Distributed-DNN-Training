#!/bin/bash
python main.py --dataset mnist -e 3 -gb 512 -tb runs/mnist/healthy
python main.py --dataset mnist -e 3 -gb 512 -tb runs/mnist/1f -f 0
python main.py --dataset mnist -e 3 -gb 512 -tb runs/mnist/1f_krum -f 0 -df pop