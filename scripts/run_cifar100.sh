#!/bin/bash
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/4w_b256_normal -w 4 -gb 256
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/4w_1f_b256 -w 4 -gb 256 -f 0
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/4w_1f_b256_correct -w 4 -f 0 -gb 256 --correct

python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_b512 -w 8 -gb 512
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_1f_b512 -w 8 -f 0 -gb 512
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_2f_b512 -w 8 -f 0 1 -gb 512
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_3f_b512 -w 8 -f 0 1 2 -gb 512
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_4f_b512 -w 8 -f 0 1 2 3 -gb 512
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_1f_b512_correct -w 8 -f 0 -gb 512 --correct
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_2f_b512_correct -w 8 -f 0 1 -gb 512 --correct
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_3f_b512_correct -w 8 -f 0 1 2 -gb 512 --correct
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_4f_b512_correct -w 8 -f 0 1 2 3 -gb 512 --correct

python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_b256 -w 8 -gb 256
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_1f_b256 -w 8 -f 0 -gb 256
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_2f_b256 -w 8 -f 0 1 -gb 256
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_3f_b256 -w 8 -f 0 1 2 -gb 256
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_4f_b256 -w 8 -f 0 1 2 3 -gb 256
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_1f_b256_correct -w 8 -f 0 -gb 256 --correct
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_2f_b256_correct -w 8 -f 0 1 -gb 256 --correct
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_3f_b256_correct -w 8 -f 0 1 2 -gb 256 --correct
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/8w_4f_b256_correct -w 8 -f 0 1 2 3 -gb 256 --correct


python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/4w_2f_b256 -w 4 -gb 256 -f 0 1
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/4w_2f_b256_correct -w 4 -gb 256 -f 0 1 --correct

python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/resnet_4w_b256_normal -w 4 -gb 256 --arch resnet18
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/resnet_4w_1f_b256 -w 4 -gb 256 --arch resnet18 -f 0
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/resnet_4w_2f_b256 -w 4 -gb 256 --arch resnet18 -f 0 1
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/resnet_4w_1f_b256_correct -w 4 -gb 256 --arch resnet18 -f 0 --correct
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/resnet_4w_2f_b256_correct -w 4 -gb 256 --arch resnet18 -f 0 1 --correct

python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/resnet_8w_b256_normal -w 4 -gb 256 --arch resnet18
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/resnet_8w_1f_b256 -w 4 -gb 256 --arch resnet18 -f 0
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/resnet_8w_2f_b256 -w 4 -gb 256 --arch resnet18 -f 0 1
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/resnet_8w_1f_b256_correct -w 4 -gb 256 --arch resnet18 -f 0 --correct
python main.py -d cifar100 -e 8 -tb runs/randm_cifar100/resnet_8w_2f_b256_correct -w 4 -gb 256 --arch resnet18 -f 0 1 --correct