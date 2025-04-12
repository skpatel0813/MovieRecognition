import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import json
from torch.cuda.amp import GradScaler, autocast

#Extracting label from filename by splitting on last hypen and converting to lowercase
def extract_label_from_filename(filename):
    return filename.rsplit('-', 1)[0].lower()

#Building a mapping between labels and indices by scanning all files in given folders
def build_global_label_map(folders):

    #Using set to avoid duplicates
    all_labels = set()
    for folder in folders:
        for f in os.listdir(folder):

            #Checking for images files in folders
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                label = extract_label_from_filename(f)
                all_labels.add(label)
    
    #Creating sorted list of labels for consistent ordering
    sorted_labels = sorted(all_labels)

    #Creating mapping dictionaries
    label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


#Creating custom dataset class for loading movie frame images
class MovieDataset(Dataset):

    def __init__(self, folder_path, label_to_idx, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.label_to_idx = label_to_idx
        self.image_paths = []
        self.labels = []

        #Populate image paths and corresponding labels
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                label = extract_label_from_filename(f)
                
                #Including labels only if they are in the mapping
                if label in label_to_idx:
                    self.image_paths.append(os.path.join(folder_path, f))
                    self.labels.append(label_to_idx[label])

    #Returning the number of images in the dataset
    def __len__(self):
        return len(self.image_paths)

    #Loading and returning one image and its label by index
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

#Defining data transformation for training set
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(10, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Defining simpler transformationg for validation set
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Building label map by scanning training directories
train_dirs = ["dataset/train/close-up", "dataset/train/wide-shot"]
label_to_idx, idx_to_label = build_global_label_map(train_dirs)

#Creating datasets for each directory and combing them
dataset1 = MovieDataset(train_dirs[0], label_to_idx, transform=train_transform)
dataset2 = MovieDataset(train_dirs[1], label_to_idx, transform=train_transform)
full_dataset = ConcatDataset([dataset1, dataset2])

#Spliting into training and validation sets (85% train, 15% validation)
val_size = int(0.15 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

#Creating data loaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#Loading EfficientNet-B0 model
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

#Replacing final classification layer for the number of classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(label_to_idx))

#Moving the model to GPU if available
model = model.to(device)

#Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = GradScaler()

#Early stopping
best_val_acc = 0
patience = 5
patience_counter = 0

print(f"Starting training on {len(full_dataset)} images across {len(label_to_idx)} classes")

#Training loop for 50 epochs max
for epoch in range(50):

    #Setting the model to training mode
    model.train()

    #Resting the metrics
    running_loss, correct, total = 0, 0, 0

    #Batch training loop
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        #Mixed precision training context
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        #Backpropagation with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #Updating the metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    #Calculating the accuracy
    train_acc = correct / total

    #Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    #Calculating the Validation accuracy
    val_acc = val_correct / val_total
    scheduler.step()

    #Printing eacb Epoach stats
    print(f"Epoch {epoch+1:02d}: Loss={running_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_efficientnet_movie_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

#Saving label mapping
with open("label_mapping.json", "w") as f:
    json.dump(idx_to_label, f)

print(f"Training complete. Best Val Accuracy: {best_val_acc:.4f}")
