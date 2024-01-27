from tqdm import tqdm
import torch
from dataset import *
from torch.utils.data import DataLoader
from model import *
from transform import *
from torchvision import transforms

lr = 0.001
batch_size = 32
epochs = 1
save_model_path = './checkpoints'
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)

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
print(f"Device : {device}")
pointnet = PointNet()
pointnet.to(device)
optimizer = torch.optim.Adam(pointnet.parameters(), lr=lr)

for epoch in range(epochs):
    pointnet.train()
    running_loss = 0.0 
    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data["pointcloud"].to(device), data["category"].to(device)
        optimizer.zero_grad()
        output, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
        loss = pointnetloss(output, labels, m3x3, m64x64)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%10 == 9 : 
            tqdm.write('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
            running_loss = 0.0

    pointnet.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(valid_loader):
            inputs, labels = data["pointcloud"].to(device), data["category"].to(device)
            output, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
            pred = torch.argmax(output, axis=1)
            correct += (pred == labels).sum().item() 
            total += labels.shape[0]
        acc = 100*(correct / total) 
        tqdm.write(f"Valid acc: {round(acc, 2)}")
        checkpoint = Path(save_model_path) / f'save_{epoch}.pthe'
        torch.save(pointnet.state_dict(), checkpoint)
        print(f"Model saved to {checkpoint}")
