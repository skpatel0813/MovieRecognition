import os
import shutil
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from PIL import Image
import json
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np

#Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Function to extract label from filename
def extract_label_from_filename(filename):
    return filename.rsplit('-', 1)[0].lower()

#Function to build a mapping beteen labels and numerical indices
def build_global_label_map(folders):
    all_labels = set()
    
    #Scanning each folder and collect all unique labels
    for folder in folders:
        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                label = extract_label_from_filename(f)
                all_labels.add(label)
    
    #Creating sorted list of labels for consistent ordering
    sorted_labels = sorted(all_labels)
    
    #Creating mapping dictionaries
    label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label

#Custom Dataset Class for loading movie frames
class MovieDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    #Returning total number of images
    def __len__(self):
        return len(self.image_paths)

    #Loading and returning one image and its label
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

#Defining image transformations for training data
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(288, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(10, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Simpler transformation for validation data
val_transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Defining all directories for training
train_dirs = ["dataset/train/close-up", "dataset/train/wide-shot", "dataset/test/medium-shot"]

#Building label mappings by scanning all images
label_to_idx, idx_to_label = build_global_label_map(train_dirs)

#Collecting all image paths and corresponding labels
all_image_paths = []
all_labels = []
for folder in train_dirs:
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            label = extract_label_from_filename(f)
            if label in label_to_idx:
                all_image_paths.append(os.path.join(folder, f))
                all_labels.append(label_to_idx[label])

#Counting the amount of images per label
label_freq = defaultdict(int)
for label in all_labels:
    label_freq[label] += 1

#Filtering out labels with less than 2 examples
filtered_paths = []
filtered_labels = []
for path, label in zip(all_image_paths, all_labels):
    if label_freq[label] >= 2:
        filtered_paths.append(path)
        filtered_labels.append(label)

#Spliting the data into training and validations sets
train_paths, test_paths, train_labels, test_labels = train_test_split(
    filtered_paths, filtered_labels, test_size=0.15, stratify=filtered_labels, random_state=42
)

#Creating a directory for the validation split and copy images there
split_test_dir = "dataset/split-test"
os.makedirs(split_test_dir, exist_ok=True)
for src in test_paths:
    shutil.copy(src, os.path.join(split_test_dir, os.path.basename(src)))

#Creating dataset objects for training and validation
train_dataset = MovieDataset(train_paths, train_labels, transform=train_transform)
val_dataset = MovieDataset(test_paths, test_labels, transform=val_transform)

#Creating DataLoaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#Initiazing EfficientNet-B2 model with pretrained weights
model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

#Modifying the classifer head for number of classes
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, len(label_to_idx))
)

#Ensuring all parameters are trainable
for param in model.parameters():
    param.requires_grad = True
model = model.to(device)

#Defining loss function with label smoothing to prevent overconfidence
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

#Using AdamW optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

#Learning rate scheduling with warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
base_scheduler = CosineAnnealingLR(optimizer, T_max=50)
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, base_scheduler], milestones=[5])

#For mixed precision training
scaler = GradScaler()

#Early stoping configuration
best_val_acc = 0
patience = 10
patience_counter = 0

print(f"Training on {len(train_dataset)} images | Validating on {len(val_dataset)} images")

#Training loop
for epoch in range(50):

    #Training phase
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        #Mixed precision training
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        #Backpropagation with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        #Updating metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    #Calculating training accuracy
    train_acc = correct / total


    #Validation phase
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    #Calculating validation accuracy
    val_acc = val_correct / val_total
    scheduler.step()


    #Printing epoch statistics
    print(f"Epoch {epoch+1:02d}: Loss={running_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    
    #Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_efficientnet_b2_movie_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

#Saving label mapping for future reference
with open("label_mapping.json", "w") as f:
    json.dump({str(v): k for k, v in label_to_idx.items()}, f)

#Final output
print(f"Training complete. Best Val Accuracy: {best_val_acc:.4f}")
print(f"Saved {len(test_paths)} test images to: {split_test_dir}")
