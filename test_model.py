import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import cv2

#Creating a custom dataset class for loading and processing test images
class MovieDataset(Dataset):
    def __init__(self, root_dir, label_mapping, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        #Creating reverse mapping from label to index
        self.label_to_idx = {v: int(k) for k, v in label_mapping.items()}
        self.idx_to_label = label_mapping

        #Collecting all image paths in directory
        all_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        #Filtering images that match the trained labels
        self.image_paths = []
        self.labels = []
        for path in all_paths:
            label = self._extract_label(path)
            if label in self.label_to_idx:
                self.image_paths.append(path)
                self.labels.append(label)

        #Converting labels to numerical indices
        self.targets = [self.label_to_idx[label] for label in self.labels]

        #Safety Check
        if len(self.image_paths) == 0:
            raise ValueError("No valid test images matched the trained labels.")

    #Extracting label from filename by splitting on last hyphen
    def _extract_label(self, path):
        filename = os.path.basename(path)
        name_part = filename.rsplit('-', 1)[0]
        return name_part.lower()

    #Returing the total number of images in dataset
    def __len__(self):
        return len(self.image_paths)

    #Loading and returning one image, its label, and path by index
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path).convert('RGB').resize((224, 224))
        if self.transform:
            image = self.transform(pil_image)
        return image, self.targets[idx], image_path

#Loading label mapping
with open("label_mapping.json", "r") as f:
    idx_to_label = json.load(f)

#Creating reverse mapping from label to index
label_to_idx = {v: int(k) for k, v in idx_to_label.items()}

#Defining image transformations (same as validation during training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Creating test dataset and loader
test_dataset = MovieDataset("dataset/test/medium-shot", label_mapping=idx_to_label, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#Loading trained model architecture
num_classes = len(label_to_idx)

#Initializing ResNet50 without pretrained weights
model = models.resnet50(weights=None)

#Replacing final fully connected layer for number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

#Loading trained weights
model.load_state_dict(torch.load("resnet_movie_model.pth", map_location=device))
model = model.to(device)
model.eval()

#Global variables to store activations and gradients
final_conv_activations = None
gradients = []

#Hook to store final convlutional layer activations
def forward_hook(module, input, output):
    global final_conv_activations
    final_conv_activations = output

#Hook to store gradients during backpropagation
def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

#Registering hooks on specific convolutional layer
model.layer4[2].conv3.register_forward_hook(forward_hook)
model.layer4[2].conv3.register_backward_hook(backward_hook)

#Initializing lists to store evaluation results
y_true, y_pred, top5_matches, csv_data = [], [], [], []

print("\nTesting model with Grad-CAM & Top-5 Accuracy...")

#Main evalutional loop
for i, (images, labels, paths) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)
    
    #Forward pass
    outputs = model(images)

    #Getting predictions
    _, preds = torch.max(outputs, 1)
    top5 = torch.topk(outputs, k=5, dim=1).indices[0].tolist()

    #Storing results
    y_true.append(labels.item())
    y_pred.append(preds.item())
    top5_matches.append(labels.item() in top5)

    #Convering indices to label names
    true_label = idx_to_label[str(labels.item())]
    pred_label = idx_to_label[str(preds.item())]
    top5_labels = [idx_to_label[str(i)] for i in top5]

    #Storing data for CSV
    csv_data.append({
        'image_path': paths[0],
        'true_label': true_label,
        'pred_label': pred_label,
        'top5': ', '.join(top5_labels)
    })

    """Hook to store final convolutional layer activations
    # Grad-CAM for first 5 samples
    if i < 5:
        model.zero_grad()
        one_hot = torch.zeros((1, outputs.size(-1)), device=device)
        one_hot[0][preds] = 1
        outputs.backward(gradient=one_hot)

        pooled_grads = torch.mean(gradients[-1], dim=[0, 2, 3])
        activation = final_conv_activations[0]
        for j in range(activation.size(0)):
            activation[j, :, :] *= pooled_grads[j]

        heatmap = torch.mean(activation, dim=0).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Convert tensor to numpy image (undo normalization)
        unnormalized = images[0].detach().cpu()
        unnormalized = unnormalized * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        unnormalized = unnormalized + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        raw_np = unnormalized.permute(1, 2, 0).numpy()
        raw_np = (raw_np * 255).clip(0, 255).astype(np.uint8)

        overlayed = cv2.addWeighted(raw_np, 0.5, heatmap, 0.5, 0)
        cam_path = f"gradcam_{i+1}_{true_label}_pred_{pred_label}.jpg"
        cv2.imwrite(cam_path, overlayed)
        print(f"Saved Grad-CAM: {cam_path}")
        """
        

#Calculating Accuracy metrics
acc = accuracy_score(y_true, y_pred)
top5_acc = sum(top5_matches) / len(top5_matches)

#Printing the results
print(f"\nTop-1 Accuracy: {acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")

#Saving the results to CSV
df = pd.DataFrame(csv_data)
df.to_csv("predictions.csv", index=False)
print("Saved predictions to predictions.csv")
