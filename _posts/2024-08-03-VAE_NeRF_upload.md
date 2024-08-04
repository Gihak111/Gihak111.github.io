---
layout: single
title:  "VAE"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 2D 사진을 3D로
단일 이미지를 VAE를 통해 당주 뷰 이미지로 전환한 후, NeRF모델을 구현하여 3D 클라우드 포인트 형태를 출력한다.  
# VAE 코드
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Load the image
image_path = r"C:\Users\연준모\2. KPASS\ver2\images\1234.jpg"
input_image = Image.open(image_path).convert('RGB')

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Apply the transformation
input_tensor = transform(input_image)  # Shape should be [3, 64, 64]

# Data augmentation
aug_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Create a dataset with repeated and augmented images
class SingleImageDataset(Dataset):
    def __init__(self, image_tensor, num_samples, transform):
        self.image_tensor = image_tensor
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.image_tensor.permute(1, 2, 0).numpy()
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = self.transform(image)
        return image

# Create DataLoader
dataset = SingleImageDataset(input_tensor, num_samples=1000, transform=aug_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define VAE
class VAE(nn.Module):
    def __init__(self, img_channels=3, z_dim=10):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * z_dim)  # for mean and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, img_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Initialize the model
vae = VAE(img_channels=3, z_dim=10)

# Define loss function and optimizer
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training loop
epochs = 10
vae.train()
for epoch in range(epochs):
    for imgs in dataloader:
        imgs = imgs.to(torch.float32)  # Ensure correct dtype
        recon_imgs, mu, logvar = vae(imgs)
        loss = loss_function(recon_imgs, imgs, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')

# 잠재 공간 조작을 통해 다양한 시점의 이미지 생성
vae.eval()
with torch.no_grad():
    # 잠재 공간에서 임의의 벡터 생성
    z = torch.randn(10, 10)  # 10개의 시점을 생성
    
    # 잠재 공간 벡터를 조작하여 다양한 시점 생성
    for i in range(10):
        z_i = z[i].unsqueeze(0)  # 단일 벡터
        generated_img = vae.decoder(z_i)
        
        # 이미지를 [0, 1] 범위에서 [0, 255] 범위로 변환
        generated_img = generated_img * 255
        generated_img = generated_img.permute(0, 2, 3, 1) # for visualization
        generated_img = generated_img.squeeze().type(torch.uint8)
        generated_img = Image.fromarray(generated_img.cpu().numpy())
        
        # 이미지를 저장하거나 출력
        generated_img.save(f'generated_view_{i}.png')

print("코드 실행은 일단 끝남")
```

위의 코드는 다음을 사용한다.  
1. 데이터 증강기법
2. 네트워크 아키텍쳐
3. 손실함수.

코드를 실행해 보면, 다중 뷰 이미지가 잘 출력됨을 확인할 수 있다.  

# NeRF 코드
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import imageio.v2 as imageio
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import trimesh

device = torch.device("cpu")  # GPU 사용을 제거하고 CPU만 사용하도록 설정

class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(6, 256)])  # 입력 크기를 6으로 조정
        self.layers.extend([nn.Linear(256, 256) for _ in range(8)])
        self.rgb_layer = nn.Linear(256, 3)
        self.density_layer = nn.Linear(256, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        rgb = torch.sigmoid(self.rgb_layer(x))
        density = torch.relu(self.density_layer(x))
        return torch.cat([rgb, density], dim=-1)

def load_images(image_dir):
    images = []
    for i in range(10):
        image_path = os.path.join(image_dir, f'generated_view_{i}.png')
        img = imageio.imread(image_path)
        images.append(img)
    return images

def preprocess_images(images):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    return [transform(img).unsqueeze(0) for img in images]

def generate_rays(h, w, K, c2w):
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dirs = torch.stack([(j - K[0, 2]) / K[0, 0], -(i - K[1, 2]) / K[1, 1], -torch.ones_like(i)], dim=-1)
    dirs = dirs.view(-1, 3)
    c2w = torch.tensor(c2w).float()  # numpy 배열을 PyTorch 텐서로 변환
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # Rotate ray directions into world space
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return torch.cat([rays_o, rays_d], dim=-1)

def train_nerf(images, K, c2w_list):
    model = NeRF().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 20  # 변경된 에포크 수
    h, w = images[0].shape[2], images[0].shape[3]

    for epoch in range(num_epochs):
        total_loss = 0
        for img, c2w in zip(images, c2w_list):
            img = img.to(device)
            rays = generate_rays(h, w, K, c2w).to(device)
            target = img.view(-1, 3).to(device)
            output = model(rays)
            output_rgb = output[:, :3]
            loss = criterion(output_rgb, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 5 == 0:  # 5 에포크마다 출력
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')
            visualize_images(images)

def visualize_images(images):
    plt.figure(figsize=(10, 10))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img.squeeze().permute(1, 2, 0).numpy())
        plt.axis('off')
    plt.show()

def generate_point_cloud(model, num_points=10000):
    rays = torch.rand((num_points, 6)).to(device)
    with torch.no_grad():
        output = model(rays)
    rgb = output[:, :3].cpu().numpy()
    points = rays[:, :3].cpu().numpy()
    return points, rgb

def save_point_cloud(points, colors, filename='point_cloud.ply'):
    point_cloud = trimesh.PointCloud(vertices=points, colors=(colors * 255).astype(np.uint8))
    point_cloud.export(filename)
    print(f'Point cloud saved to {filename}')

def visualize_point_cloud(file_path):
    point_cloud = trimesh.load(file_path)
    points = point_cloud.vertices
    colors = point_cloud.colors / 255.0

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Point Cloud Visualization')
    plt.show()

def main():
    image_dir = r'C:\Users\연준모\2. KPASS\ver2'
    print("코드 실행 시작")
    images = load_images(image_dir)
    images = preprocess_images(images)

    # 예시 카메라 매트릭스와 포즈 설정 (사용자 설정 필요)
    K = np.array([[256, 0, 128], [0, 256, 128], [0, 0, 1]], dtype=np.float32)
    c2w_list = [np.eye(4) for _ in range(10)]  # 각 이미지에 대한 카메라 포즈

    train_nerf(images, K, c2w_list)

    model = NeRF().to(device)
    points, colors = generate_point_cloud(model)
    point_cloud_path = os.path.join(image_dir, 'point_cloud.ply')
    save_point_cloud(points, colors, point_cloud_path)

    visualize_point_cloud(point_cloud_path)

if __name__ == "__main__":
    main()

```
이어서, NeRF 코드 이다.  
1. NeRF 아키텍켜의 3D 좌표에서 RGB 및 밀도 예측을 가능하게 했다.  
2. 카메라 매트릭스와 포즐르 추가해 빈 공간을 더 잘 초래한다.  

이 NeRF 코들르 통해 앞서 VAE로 만든 다중뷰 이미지를 3D 모델로 만드는 멀티 모델 구조를 만들 수 있다.  