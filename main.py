from dataloader.mnist import load_mnist

BATCH_SIZE = 20

train_loader, test_loader = load_mnist(BATCH_SIZE)