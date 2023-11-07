# Updates & Notes

**[Notes]**
- Work under a dev branch, and then merge to the master branch later. 

**[TODOs]**
- [ ] Implement the detection algorithms
- [ ] Implement the alternative DDP algorithms (pass in sub data-batch replicas)
- [ ] Implement the correction algorithms (data collection)
- [ ] Implement the correction algorithms (gradient correction model)
- [ ] Implement the correction algorithms (run in a saparate thread)

<br>

**[11-07-2023]**
<details>
  <summary> updates </summary>

- Now have a DDP simulator that suports single and multiple processes. 
- Able to simulate the DDP process for MNIST and CIFAR100 dataset with different models.

</details>

**[10-19-2023]**
<details>
  <summary> updates  </summary>

- [Test] Modulized the model and dataloader. Many models are added to the model.py file to support classification tasks. And the current dataloader only a dummy dataloader which generates data for a given function. 

</details>

**[10-13-2023]**
<details>
  <summary> ZY's updates  </summary>

- I wrote a simple script that you can run on you laptop, simulating DDP. The logic is pretty simple, you could look through the code. 
- I realized there could be two types of faulty workers: 
    1. [Noisy workers] Data batch ran through the model, noise added to the gradients after. Possible for correction. 
    2. [Byzantine workers] Data batch did NOT ran through the model, the gradients are stright up random. Impossible to correct.
- So our proposal is not very clear about the "correction". I think the paper is trying to say that the server should be able to detect the faulty workers, and then correct the gradients. But I think it's impossible to correct the gradients if the workers are byzantine. 
</details>