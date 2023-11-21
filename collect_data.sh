#!/bin/bash

# deterministic
python main.py -d mnist -f 0 --proc 4 --data_coll_path data/grad_datasets/determ/mnist_cusMLP.pkl

python main.py -d cifar10 -f 0 --proc 4 --data_coll_path data/grad_datasets/determ/cifar10_cusCNN.pkl
python main.py -d cifar10 -f 0 --proc 4 --arch resnet18 --data_coll_path data/grad_datasets/determ/cifar10_resnet18.pkl

python main.py -d cifar100 -f 0 --proc 4 --data_coll_path data/grad_datasets/determ/cifar100_cusCNN.pkl
python main.py -d cifar100 -f 0 --proc 4 --arch resnet18 --data_coll_path data/grad_datasets/determ/cifar100_resnet18.pkl