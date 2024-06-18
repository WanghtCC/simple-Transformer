import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.vit_model import ViT
from dataloader.mnist import load_mnist

BATCH_SIZE = 20
EPOCH = 5

torch.manual_seed(123)
train_loader, test_loader = load_mnist(BATCH_SIZE)

model = ViT(
    image_size=28,
    patch_size=7,
    num_classes=10,
    channels=1,
    dim=64,
    depth=6,
    heads=8,
    mlp_dim=128,
)
optimizer = optim.Adam(model.parameters(), lr=0.0003)

start_time = time.time()
train_loss_list, test_loss_list = [], []
for epoch in range(1, EPOCH + 1):
    epoch_time = time.time()
    model.train()
    for i, (img, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(img)
        output = F.log_softmax(output, dim=1)
        # loss = F.cross_entropy(output, label)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(
                "[ "
                + "{:5}".format(i * len(img))
                + "/"
                + "{:5}".format(len(train_loader.dataset))
                + " ("
                + "{:3.0f}".format(100 * i / len(train_loader))
                + "%)] Loss: "
                + "{:6.4f}".format(loss.item())
            )
            train_loss_list.append(loss.item())

    print("Training time: " + "{:5.1f}".format(time.time() - epoch_time) + "s")

    model.eval()
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for img, label in test_loader:
            output = model(img)
            output = F.log_softmax(output, dim=1)
            # loss = F.cross_entropy(output, label)
            loss = F.nll_loss(output, label)
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(label).sum()

    avg_loss = total_loss / len(test_loader)
    test_loss_list.append(avg_loss)
    accuracy = correct_samples.float() / len(test_loader.dataset)
    print(
        "Test loss: "
        + "{:.4f}".format(avg_loss)
        + " Accuracy: "
        + "{:4.2f}".format(accuracy)
    )

    print(
        "Epoch: "
        + str(epoch)
        + " Time: "
        + "{:5.1f}".format(time.time() - epoch_time)
        + "s"
    )
