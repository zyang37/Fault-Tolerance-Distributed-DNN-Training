# Fault-Tolerance-Distributed-DNN-Training
Simulating DDP training with the presence of malicious workers that inject trivial noise into their local gradients during each iteration. 

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python main.py -h

usage: main.py [-h] [-d] [--arch] [-e] [-gb] [-w] [-f  [...]] [-df] [--proc] [-tb] [--device] [--correct] [--data_coll_path]

DDP simulation

optional arguments:
  -h, --help            show this help message and exit
  -d , --dataset        Dataset to use
  --arch                Model to use
  -e , --epoch          Number of epochs
  -gb , --global_batch_size 
                        Global batch size
  -w , --worker         Number of workers/sub-batches (Note: global batch size must be divisible by number of workers))
  -f  [ ...], --faulty  [ ...]
                        Indics of faulty worker (Ex: -f 0 1 2)
  -df , --defense       Defense method
  --proc                Number of processes
  -tb , --tb            Tensorboard log directory
  --device              Device to use
  --correct             Whether to use error correction
  --data_coll_path      Path to data collection
```

## Example
Running DDP training with 16 workers, 5 of which are malicious, on MNIST dataset with error correction.

```
python main.py -d mnist -w 16 -f 0 1 2 3 4 --correct

Device: cpu
Number of workers: 16
Number of processes: 1
Faulty worker idxs: [0, 1, 2, 3, 4]
Namespace(dataset='mnist', arch=None, epoch=3, global_batch_size=512, worker=16, faulty=[0, 1, 2, 3, 4], defense=None, proc=1, tb=None, device='cpu', correct=True, data_coll_path=None)

Worker-to-dataBatch assignments
W-B MAP: {0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15}
[AGG] Init correction model for worker 0
[AGG] Init correction model for worker 1
[AGG] Init correction model for worker 2
[AGG] Init correction model for worker 3
[AGG] Init correction model for worker 4
[CORRECTION] Training ...
[CORRECTION] Training loss (MSE, MAPE): 6.644912e-07 0.0004825247
[CORRECTION] Training ...
[CORRECTION] Training loss (MSE, MAPE): 6.644768e-07 0.00048268036
[CORRECTION] Training ...
[CORRECTION] Training loss (MSE, MAPE): 6.644902e-07 0.0004825253
[CORRECTION] Training ...
[CORRECTION] Training loss (MSE, MAPE): 6.6449115e-07 0.00048252486
[CORRECTION] Training ...
[CORRECTION] Training loss (MSE, MAPE): 6.644911e-07 0.00048252384
EP: 1/3, sub-batch: 1/118, avg sub-batch loss: 2.295
W-B MAP: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15}
EP: 1/3, sub-batch: 2/118, avg sub-batch loss: 2.312
W-B MAP: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15}
EP: 1/3, sub-batch: 3/118, avg sub-batch loss: 2.306
W-B MAP: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15}
EP: 1/3, sub-batch: 4/118, avg sub-batch loss: 2.308
W-B MAP: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15}
EP: 1/3, sub-batch: 5/118, avg sub-batch loss: 2.298
W-B MAP: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15}
...
```

## Visualization
```
tensorboard --logdir=runs
```
