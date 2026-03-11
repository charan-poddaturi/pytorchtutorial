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


# datatyes

#finding datatype
x.dtype

# assigning datatype
torch.tensor([1.0,2.0,3.0],dtype=torch.int32)
torch.tensor([1,2,3],dtype=torch.float64)


# using to()
x.to(torch.float32)


# we can do all scalar operations with tensors

# we can also add two tensors with same shapes

# for absolute values
torch.abs(x)

# fro negative values 
torch.neg(x)

# for  rounded values
torch.round(x)

# for ceil values
torch.ceil(x)

# for floor values
torch.floor(x)

# for clamp(values under 2 becomes 2 and values above 3 becomes 3)
torch.clamp(x, min=2, max=3)

