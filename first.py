import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
# using empty
a=torch.empty(2,3)

# checking type
type(a)

# using zeros(all values initalized to zero)
torch.zeros(2,3)

# using ones(all values are initialized to 1)
torch.ones(2,3)

# using random values(these values could change if run again)
torch.rand(2,3)

# using seed(these values doesn't change even is we run multiple times)
torch.seed(2,3)

# using manual_seed
torch.manual_seed(2,3)

# using tensor(we could make custom tensors)
torch.tensor([[1,2,3],[4,5,6]])

# other ways

# arange
print("using arange ->", torch.arange(0,10,2))

# using linspace
print("using linspace ->", torch.linspace(0,10,10))

# using eye
print("using eye ->", torch.eye(5))

# using full
print("using full ->", torch.full((3, 3), 5))