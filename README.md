# Updates & Notes

**[10-13-2023]**
- I wrote a simple script that you can run on you laptop, simulating DDP. The logic is pretty simple, you could look through the code. 
- I realized there could be two types of faulty workers: 
    1. [Noisy workers] Data batch ran through the model, noise added to the gradients after. Possible for correction. 
    2. [Byzantine workers] Data batch did NOT ran through the model, the gradients are stright up random. Impossible to correct.
- So our proposal is not very clear about the "correction". I think the paper is trying to say that the server should be able to detect the faulty workers, and then correct the gradients. But I think it's impossible to correct the gradients if the workers are byzantine. 