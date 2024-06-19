import time
import torch
import datetime
import torch.optim as optim
import torch.nn.functional as F
from models.vit_model import ViT
from dataloader.mnist import load_mnist

BATCH_SIZE = 20
EPOCH = 5
LR = 1e-4

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
optimizer = optim.Adam(model.parameters(), lr=LR)

start_time = time.time()
train_loss_list, test_loss_list = [], []
for epoch in range(1, EPOCH + 1):
    epoch_start_time = time.time()
    count = 0
    sum_loss = 0
    correct_samples = 0
    acc = 0
    model.train()
    for i, (img, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(img)
        output = F.log_softmax(output, dim=1)
        # loss = F.cross_entropy(output, label)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        count += BATCH_SIZE

        _, pred = torch.max(output, dim=1)
        correct_samples += pred.eq(label).sum()

        acc = correct_samples.item() / count
        
        if i % 100 == 0 and i:
            print(
                f'epoch: {epoch} ' 
                + f'[ {i * len(img):5}' + f'/{len(train_loader.dataset):5}'
                + f' ({100 * i / len(train_loader):3.0f}' + '%)] | '
                + f'loss: {loss.item():6.4f} | '
                + f'correct: {pred.eq(label).sum() / BATCH_SIZE:.3f}'
            )
            train_loss_list.append(loss.item())

    print(f'epoch: {epoch} ### lr: {optimizer.param_groups[0]["lr"]:.6f} ### loss: {sum_loss / count:.4f} ### acc: {acc:1.3f}')
    elapsed = round(time.time() - epoch_start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Training Finished. Total elapsed time(h:m:s): {}'.format(elapsed))
    print('-' * 80)

    model.eval()
    correct_samples = 0
    total_loss = 0
    eval_start_time = time.time()
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
    print(f'eval:----------> loss: {avg_loss:6.4f} ### acc: {accuracy:1.3f}')

    elapsed = round(time.time() - eval_start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))

    print('Eval Finished. Total elapsed time(h:m:s): {}'.format(elapsed))
    print('-' * 80)

elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))
print('Train Finished. Total elapsed time(h:m:s): {}'.format(elapsed))
print('-' * 80)
