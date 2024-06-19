import torch
import torchvision

def load_mnist(batch_size_train, batch_size_test=1):
    transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform_mnist)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    test_set = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform_mnist)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)
    
    print('MNIST dataset load successful')
    print('-' * 80)
    print(f'Total train image-label pair: {len(train_loader.dataset)}')
    print(f'Total test  image-label pair: {len(test_loader.dataset)}')
    print('-' * 80)

    return train_loader, test_loader
