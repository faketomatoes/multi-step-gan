import torch

x = torch.ones((2, 2))
print(torch.sub(1, x))
print(x.size()[0])
index = torch.tensor([1], dtype=torch.int64)
x[0].index_fill_(0, index, 0)
print(x)
y = torch.tensor(0)
z = torch.ones((1))
print(z.item())
print(y)