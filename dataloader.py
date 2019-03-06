import torch
from torchvision import datasets,transforms

def data_loader(batch_size):
    root='data/'
    train_loader=torch.utils.data.DataLoader(
        datasets.MNIST(root,train=True,download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
            ])),
        batch_size=batch_size,
        shuffle=True)
    test_loader=torch.utils.data.DataLoader(
        datasets.MNIST(root,train=False,download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])),
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader,test_loader