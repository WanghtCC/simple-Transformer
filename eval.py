import time
import torch
import datetime
import torch.optim as optim
import torch.nn.functional as F
from models.vit_model import ViT
from models.init import init_model
from dataloader.mnist import load_mnist
from utils import load_pretrained_weights, print_img

IMG_IDX = 0
BATCH_SIZE = 128

_, test_loader = load_mnist(BATCH_SIZE)

model = init_model('vit',
                   image_size=28,
                   patch_size=14,
                   num_classes=10,
                   channels=1,
                   dim=128,
                   depth=5,
                   heads=8,
                   mlp_dim=256,
)

load_pretrained_weights(model, torch.load('./result/best_model.pth'))

data = test_loader.dataset.data[IMG_IDX].unsqueeze(0).unsqueeze(0)
label = test_loader.dataset.targets[IMG_IDX].unsqueeze(0)
img = data / 1.

# print(img.shape)
# print(label.shape)

model.eval()
with torch.no_grad():
    output = model(img)
    output = F.log_softmax(output, dim=1)
    _, pred = torch.max(output, 1)

print_img(data)
print(f'predict: {pred.item()}, label: {label.item()}')
