#!/bin/bash
python main.py --dataset mnist -gb 1024 -tb runs/mnist/healthy
python main.py --dataset mnist -gb 1024 -tb runs/mnist/1f -f 0
python main.py --dataset mnist -gb 1024 -tb runs/mnist/1f_rm -f 0 -df pop