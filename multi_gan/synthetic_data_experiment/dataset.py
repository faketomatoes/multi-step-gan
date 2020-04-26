import math
import torch
from torch.utils.data import Dataset


class SyntheticData(Dataset):
    def __init__(self, size=10000, radius=1, std_err=0.1):
        super(SyntheticData, self).__init__()
        self.data = torch.zeros((size, 3))
        for k in range(size):
            theta = 2 * math.pi * torch.rand((1))
            self.data[k, 0] = theta
            noise = torch.randn(1, 2)
            mean = torch.tensor([radius * torch.cos(theta).item(), radius * torch.sin(theta).item()])
            noise = mean + std_err * noise
            self.data[k, 1:] = noise
        self.std_err = std_err
        self.radius = radius
        return
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

