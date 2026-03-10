import torch

x = torch.tensor([[1,2,3,],[4,5,6]])
print(x)

x.shape()

#creates and empty tensor with the shape of x
torch.empty_like(x) 

# similarly
torch.zeros_like(x)
torch.ones_like(x)

# this gives an error because x has integers but rand functions gives the outputs as float which are between 0 1nad 1
torch.rand_like(x) 