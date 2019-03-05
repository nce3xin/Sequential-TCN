import torch
from torchvision import datasets,transforms

def data_loader(bs):
    train_loader=torch.utils.data.DataLoader(
        datasets.MNIST('data/',train=True,download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.1307,0.3081)
            ])),
        batch_size=bs,
        shuffle=True)
    test_loader=torch.utils.data.DataLoader(
        datasets.MNIST('data/',train=False,download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307,0.3081)
        ])),
        batch_size=bs,
        shuffle=True
    )
    return train_loader,test_loader