import torch

print(torch.__version__)
x = torch.rand(5, 3) # create a 5x3 matrix of random numbers
print(x)

import numpy as np

numbers = np.random.rand(5, 3) # create a 5x3 matrix of random numbers

print(numbers)

numbers = np.random.randn(5, 3) # create a 5x3 matrix of random numbers from a normal distribution
print(numbers.shape)