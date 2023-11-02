#!/bin/bash
python main.py --dataset cifar100 -e 3 -gb 512 -tb runs/cifar100/healthy
python main.py --dataset cifar100 -e 3 -gb 512 -tb runs/cifar100/1f -f 0
python main.py --dataset cifar100 -e 3 -gb 512 -tb runs/cifar100/1f_rm -f 0 -df mean
# python main.py --dataset cifar100 -e 3 -gb 512 -tb runs/cifar100/1f_pop -f 0 -df pop