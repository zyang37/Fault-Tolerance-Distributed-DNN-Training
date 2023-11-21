#!/bin/bash
# python main.py --dataset cifar100 -e 3 -gb 512 -tb runs/cifar100/healthy
# python main.py --dataset cifar100 -e 3 -gb 512 -tb runs/cifar100/1f -f 0
# python main.py --dataset cifar100 -e 3 -gb 512 -tb runs/cifar100/1f_rm -f 0 -df mean
# python main.py --dataset cifar100 -e 3 -gb 512 -tb runs/cifar100/1f_pop -f 0 -df pop

python main.py -d cifar100 -e 8 -tb runs/cifar100_4w_b2048_correct -w 4 -gb 2048
python main.py -d cifar100 -e 8 -tb runs/cifar100_4w_1f_b2048_correct -w 4 -f 0 -gb 2048
python main.py -d cifar100 -e 8 -tb runs/cifar100_4w_1f_b2048_correct -w 4 -f 0 -gb 2048 --correct

python main.py -d cifar100 -e 8 -tb runs/cifar100_8w_b2048 -w 8 -gb 2048
python main.py -d cifar100 -e 8 -tb runs/cifar100_8w_1f_b2048 -w 8 -f 0 -gb 2048
python main.py -d cifar100 -e 8 -tb runs/cifar100_8w_2f_b2048 -w 8 -f 0 1 -gb 2048
python main.py -d cifar100 -e 8 -tb runs/cifar100_8w_3f_b2048 -w 8 -f 0 1 2 -gb 2048
python main.py -d cifar100 -e 8 -tb runs/cifar100_8w_4f_b2048 -w 8 -f 0 1 2 3 -gb 2048
python main.py -d cifar100 -e 8 -tb runs/cifar100_8w_1f_b2048_correct -w 8 -f 0 -gb 2048 --correct
python main.py -d cifar100 -e 8 -tb runs/cifar100_8w_2f_b2048_correct -w 8 -f 0 1 -gb 2048 --correct
python main.py -d cifar100 -e 8 -tb runs/cifar100_8w_3f_b2048_correct -w 8 -f 0 1 2 -gb 2048 --correct
python main.py -d cifar100 -e 8 -tb runs/cifar100_8w_4f_b2048_correct -w 8 -f 0 1 2 3 -gb 2048 --correct
