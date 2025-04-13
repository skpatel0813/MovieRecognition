import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from sklearn.metrics import accuracy_score
import pandas as pd

#Setting the device to GPU if avaiable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Custom Dataset Class for loading movie frames
class MovieDataset(Dataset):
    def __init__(self, root_dir, label_mapping, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        #Creating reverse mapping from label to index
        self.label_to_idx = {v: int(k) for k, v in label_mapping.items()}
        self.idx_to_label = label_mapping

        #Getting all image paths in directory
        all_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        #Filtering images that match trained labels
        self.image_paths = []
        self.labels = []
        for path in all_paths:
            label = self._extract_label(path)
            if label in self.label_to_idx:
                self.image_paths.append(path)
                self.labels.append(self.label_to_idx[label])

        #Alias for labels
        self.targets = self.labels

        #Safety Check
        if len(self.image_paths) == 0:
            raise ValueError("No valid test images matched the trained labels.")

    def _extract_label(self, path):
        filename = os.path.basename(path)
        return filename.rsplit('-', 1)[0].lower()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB').resize((288, 288))
        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx], image_path

#Load label mapping created during training
with open("label_mapping.json", "r") as f:
    idx_to_label = json.load(f)

#Creating reverse mapping from label to index
label_to_idx = {v: int(k) for k, v in idx_to_label.items()}

#Defining image transframtions
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Creating test dataset and loader
test_dataset = MovieDataset("dataset/split-test", label_mapping=idx_to_label, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#Initializing EfficientNet-B2 model architecture
model = efficientnet_b2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, len(label_to_idx))
)

#Loading trained weights
model.load_state_dict(torch.load("best_efficientnet_b2_movie_model.pth", map_location=device))
model = model.to(device)
model.eval()

#Initializing lists to store evaluation results
y_true, y_pred, top5_matches, csv_data = [], [], [], []

print("\nEvaluating model on split-test set...")

#Evaluation loop
for i, (images, labels, paths) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)

    _, preds = torch.max(outputs, 1)
    top5 = torch.topk(outputs, k=5, dim=1).indices[0].tolist()

    y_true.append(labels.item())
    y_pred.append(preds.item())
    top5_matches.append(labels.item() in top5)

    true_label = idx_to_label[str(labels.item())]
    pred_label = idx_to_label[str(preds.item())]
    top5_labels = [idx_to_label[str(i)] for i in top5]

    csv_data.append({
        'image_path': paths[0],
        'true_label': true_label,
        'pred_label': pred_label,
        'top5': ', '.join(top5_labels)
    })

#Calculating accuracy metrics
acc = accuracy_score(y_true, y_pred)
top5_acc = sum(top5_matches) / len(top5_matches)

#Printing results
print(f"\nTop-1 Accuracy: {acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")

#Saving predictions to CSV
df = pd.DataFrame(csv_data)
df.to_csv("split_test_predictions.csv", index=False)
print("Saved predictions to split_test_predictions.csv")
