import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
class Config:
    lr = 0.001
    batch_size = 64
    momentum = 0.9
    log_interval = 100
    test_interval = 10
    epochs = 60
    weight_decay = 1e-3
    cuda_device = '0'
    imageSize = 64
    data_root = "~/datasets"
    model_save_root = "./trained-models"
cfg = Config()

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = list(hidden_layers)
        layers = []
        current_dim = input_dim
        for hiddens in hidden_layers:
            layers.append(nn.Linear(current_dim, hiddens))
            layers.append(nn.ReLU())
            current_dim = hiddens
        layers.append(nn.Linear(current_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        input = input.view(-1, self.input_dim)
        output = self.model(input)
        return output

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_device
    dataset_train = torchvision.datasets.MNIST(root=cfg.data_root, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(cfg.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
    dataset_test = torchvision.datasets.MNIST(root=cfg.data_root, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(cfg.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]), train=False)
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=cfg.batch_size, shuffle=True)
    model = MLP(cfg.imageSize * cfg.imageSize, 10, [1024, 1024, 1024]).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    for i in range(cfg.epochs):
        print("epoch {}:".format(i + 1))
        k = 0
        num_sample = 0
        current_loss = 0
        current_correct = 0
        for (x, y) in dataloader_train:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            Loss = criterion(output, y)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            pred = output.detach().max(1)[1]
            current_correct += pred.cpu().eq(y.cpu()).sum().item()
            current_loss += Loss.cpu().detach()
            k += 1
            num_sample += len(y)
            if (k + 1) % cfg.log_interval == 0:
                avg_loss = current_loss / num_sample
                accuracy = current_correct / num_sample
                print("iteration{}:acc:{},avg_loss:{}".format(k + 1, accuracy, avg_loss))
        if (i + 1) % cfg.test_interval == 0:
            current_correct = 0
            num_sample = 0
            for (x, y) in dataloader_test:
                x = x.cuda()
                y = y.cuda()
                output = model(x)
                pred = output.detach().max(1)[1]
                current_correct += pred.cpu().eq(y.cpu()).sum().item()
                num_sample += len(y)
            accuracy = current_correct / num_sample
            print("test acc:{}".format(accuracy))
    model = model.cpu()
    torch.save(model.state_dict(), os.path.join(cfg.model_save_root, "MNIST_classifier.pth"))
    return

if __name__ == '__main__':
    train()
    

