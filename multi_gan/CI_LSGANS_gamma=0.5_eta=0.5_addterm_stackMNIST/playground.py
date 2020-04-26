import torchvision.datasets
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

dataset = torchvision.datasets.MNIST(root="~/datasets/", download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                               ]))
print(len(dataset))
dataloader1 = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)
dataloader2 = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)
dataloader3 = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)
G = torch.zeros((0, 3, 28, 28))
Y = torch.zeros((0), dtype=torch.int64)

for (x1, y1), (x2, y2), (x3, y3) in zip(dataloader1, dataloader2, dataloader3):
    x = torch.cat((x1, x2, x3), dim = 1)
    print(x.size())
    xR = x[:, 0, :, :]
    print(xR.size())
    z = torch.cat((x, G), dim=0)
    print(z.size())
    Y = torch.cat((Y, y1), dim = 0)
    print(Y.size())
    vutils.save_image(x, "ali.png", nrow=2)
    break

Z = np.zeros(100)
print(Z)