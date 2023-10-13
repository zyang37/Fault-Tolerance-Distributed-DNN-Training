import torch
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import Process, Manager

# Define model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# Worker process
def worker(model, data, gradients_list, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = ((output - 5)**2).mean()
    loss.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    gradients_list.append(gradients)

if __name__ == '__main__':
    manager = Manager()

    # Initialize global model and optimizer
    global_model = SimpleModel()
    optimizer = optim.SGD(global_model.parameters(), lr=0.01)

    n_epochs = 3  # Number of epochs
    n_iters = 4  # Number of iterations per epoch

    for epoch in range(n_epochs):
        for iteration in range(n_iters):
            gradients_list = manager.list()

            # Create worker data batches
            data_batches = [torch.randn(5, 10) for _ in range(2)]

            processes = []
            for data_batch in data_batches:
                p = Process(target=worker, args=(global_model, data_batch, gradients_list, optimizer))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # Aggregate gradients
            aggregated_gradients = [sum(grad) for grad in zip(*gradients_list)]

            # Update global model
            optimizer.zero_grad()
            for p, agg_grad in zip(global_model.parameters(), aggregated_gradients):
                p.grad = agg_grad
            optimizer.step()

            print(f'Epoch: {epoch+1}, Iteration: {iteration+1}')
            