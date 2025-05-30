# -*- coding: utf-8 -*-
"""XAI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1W9Ji_NW-_JHilcFozp95DM2cd3W_09fi
"""

!pip install torch torchvision matplotlib scikit-learn



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

from google.colab import drive
drive.mount('/content/drive')

zip_path = '/content/drive/MyDrive/brain-tumor-mri-dataset.zip'

!unzip "{zip_path}" -d /content/

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(root='/content/Training', transform=transform)
test_dataset = datasets.ImageFolder(root='/content/Testing', transform=transform)

print(f'Training images: {len(train_dataset)}')
print(f'Testing images: {len(test_dataset)}')

!ls /content/Training

from torch.utils.data import random_split

# Split the training dataset (80% for training, 20% for validation)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_data, val_data = random_split(train_dataset, [train_size, val_size])

print(f'Training data size: {len(train_data)}')
print(f'Validation data size: {len(val_data)}')

from torch.utils.data import DataLoader

batch_size = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Max Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 28 * 28)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Initialize the model
model = BrainTumorCNN().cuda()

# Loss function
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()


        train_accuracy = 100 * correct / total
        val_accuracy = 100 * val_correct / val_total
        print(f'Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - '
              f'Train Accuracy: {train_accuracy:.2f}% - Val Accuracy: {val_accuracy:.2f}%')


train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

evaluate_model(model, test_loader)

# Function to register hooks
gradients = None
activations = None

def save_activations_hook(module, input, output):
    global activations
    activations = output

def save_gradients_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Register hooks on the last conv layer
last_conv_layer = model.conv3
last_conv_layer.register_forward_hook(save_activations_hook)
last_conv_layer.register_backward_hook(save_gradients_hook)

import numpy as np
import cv2
import torch

def generate_gradcam(model, input_image, class_idx=None):
    model.eval()
    device = next(model.parameters()).device

    input_image = input_image.to(device)  # Should be [1, 3, 224, 224]
    output = model(input_image)

    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()

    model.zero_grad()
    output[:, class_idx].backward()

    grads = gradients.detach().cpu().numpy()[0]       # [C, H, W]
    acts = activations.detach().cpu().numpy()[0]      # [C, H, W]

    weights = np.mean(grads, axis=(1, 2))
    gradcam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        gradcam += w * acts[i]

    gradcam = np.maximum(gradcam, 0)
    gradcam = cv2.resize(gradcam, (224, 224))
    gradcam -= gradcam.min()
    gradcam /= gradcam.max()

    return gradcam

import matplotlib.pyplot as plt

def display_gradcam(input_tensor, gradcam):
    if input_tensor.shape[0] == 1:  # Grayscale to RGB
        input_tensor = input_tensor.repeat(3, 1, 1)

    input_image = input_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    input_image = np.uint8((input_image - input_image.min()) / (input_image.max() - input_image.min()) * 255)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(input_image, alpha=0.5)
    plt.imshow(gradcam, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis('off')

    plt.show()

# Save the trained model
torch.save(model.state_dict(), 'brain_tumor_cnn.pth')
print("Model saved successfully!")

# Load the model
model = BrainTumorCNN().cuda()
model.load_state_dict(torch.load('brain_tumor_cnn.pth'))
model.eval()  # Set the model to evaluation mode

from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load image
img_path = 'test/test2.jpg'
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).cuda()

print(f"Image shape: {input_tensor.shape}")

# Predict
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)

print(f"Predicted class: {predicted_class.item()}")

# Generate Grad-CAM
gradcam = generate_gradcam(model, input_tensor, class_idx=predicted_class.item())

# Display
display_gradcam(input_tensor.squeeze(0), gradcam)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model_with_metrics(model, test_loader, class_names):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Print classification report
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))

    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Class names for brain tumor dataset
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Evaluate the model
evaluate_model_with_metrics(model, test_loader, class_names)