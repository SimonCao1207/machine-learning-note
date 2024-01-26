from tqdm import tqdm
import torch
from dataset import *
from torch.utils.data import DataLoader
from model import *
from transform import *
from torchvision import transforms

lr = 0.001
batch_size = 32
epochs = 15
save_model_path = './checkpoints'

train_transforms = transforms.Compose([
    PointSampler(1024),
    Normalize(),
    RandRotation_z(),
    RandomNoise(),
    ToTensor()
])
train_ds = ModelNetDataset(MODELNET10_PATH, transform=train_transforms)
valid_ds = ModelNetDataset(MODELNET10_PATH, split='test', transform=train_transforms)

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=batch_size*2)

print('Train dataset size: ', len(train_ds))
print('Valid dataset size: ', len(valid_ds))
print('Number of classes: ', len(train_ds.classes))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pointnet = PointNet()
pointnet.to(device)
optimizer = torch.optim.Adam(pointnet.parameters(), lr=lr)

for epoch in tqdm(range(epochs)):
    pointnet.train()
    running_loss = 0.0 
    for i, data in enumerate(train_loader):
        inputs, labels = data["pointcloud"].to(device), data["category"].to(device)
        optimizer.zero_grad()
        output, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
        break
    break