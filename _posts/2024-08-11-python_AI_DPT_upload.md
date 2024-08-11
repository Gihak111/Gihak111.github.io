---
layout: single
title:  "DTP로 깊이 측정하기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# DPTDepthModel
인푹으로 받는 사진의 깊이를 축정하고, 이를 통해 3D로 출력해 주는 모델이다.  
이것 저것 도정할 것이 많아서, 완벽하게 3D 모델로 전환되지는 않는다.  
애초에 이 기술을, 드론 같은 걸로 등고선을 만들때 사용하며, 3D 모델링 용도로 사용할 수 있는지, 없는지를 확인하기 위해 시도해 봤다.   


# 코드
```python
# app.py
import os
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from midas.dpt_depth import DPTDepthModel
import numpy as np
import base64
from io import BytesIO

# Load pre-trained model
model = DPTDepthModel(path="dpt_large-midas-2f21e586.pt", backbone="vitl16_384", non_negative=True)
model.eval()

# Transform input image
transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    image = Image.open(file.stream)
    input_image = transform(image).unsqueeze(0)

    with torch.no_grad():
        depth = model(input_image)

    depth = depth.squeeze().cpu().numpy()

    # Convert depth map to image
    depth_img = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
    depth_img = depth_img.astype(np.uint8)
    depth_pil = Image.fromarray(depth_img)

    # Convert image to base64
    buffered = BytesIO()
    depth_pil.save(buffered, format="PNG")
    depth_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'depth_map': depth_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

OpenCV와 matplotlib
```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from midas.dpt_depth import DPTDepthModel

# Load pre-trained model
model = DPTDepthModel(path="dpt_large-midas-2f21e586.pt", backbone="vitl16_384", non_negative=True)
model.eval()

# Transform input image
transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# Load and preprocess image
image_path = r"C:\Users\연준모\KPASS\ver0.00\IMG_1391.jpeg"
image = Image.open(image_path)
input_image = transform(image).unsqueeze(0)

# Predict depth
with torch.no_grad():
    depth = model(input_image)

# Convert to numpy for visualization
depth = depth.squeeze().cpu().numpy()

# Visualize the depth map as 2D image
plt.imshow(depth, cmap='inferno')
plt.colorbar()
plt.show()

# Prepare 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for 3D plotting
h, w = depth.shape
x = np.linspace(0, w - 1, w)
y = np.linspace(0, h - 1, h)
x, y = np.meshgrid(x, y)
z = depth

# Plot surface
ax.plot_surface(x, y, z, cmap='inferno', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth')

# Show plot
plt.show()
```

다음과 같은 결과가 출력된다.  
![결과](https://github.com/user-attachments/assets/240ad0d7-4b71-4c9d-b73b-08bb20913b00)  