import torch
from torchvision.utils import save_image
import os
from datetime import datetime

# Load model once
model = torch.load('model/my_model.pth', map_location=torch.device('cpu'))
model.eval()

def generate_digit_images(digit, count=5):
    os.makedirs('generated', exist_ok=True)
    images = []
    for i in range(count):
        z = torch.randn(1, 100)  # adjust size if needed
        # label = torch.tensor([digit])  # use if your model is conditional

        generated = model(z)  # or model(z, label) if conditional
        filename = f'generated/{digit}_{i}_{datetime.now().timestamp()}.png'
        save_image(generated, filename)
        images.append(filename)
    return images
