import os
import shutil
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

project_dir = r"C:\Users\aboud\Documents\PlantDiseaseProject"
raw_dataset = r"C:\Users\aboud\Downloads\plantvillage"
train_dir = os.path.join(project_dir, "dataset", "train")
test_dir = os.path.join(project_dir, "dataset", "test")
unknown_dir = os.path.join(project_dir, "unknown")
sorted_dir = os.path.join(project_dir, "sorted")
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sort_dataset(source_dir, dest_dir):
    healthy = os.path.join(dest_dir, "train", "healthy")
    diseased = os.path.join(dest_dir, "train", "diseased")
    os.makedirs(healthy, exist_ok=True)
    os.makedirs(diseased, exist_ok=True)

    for folder in os.listdir(source_dir):
        label = "healthy" if "healthy" in folder.lower() else "diseased"
        src_folder = os.path.join(source_dir, folder)
        dst_folder = os.path.join(dest_dir, "train", label)

        for file in os.listdir(src_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(src_folder, file)
                dst_path = os.path.join(dst_folder, f"{folder}_{file}")
                try:
                    shutil.copyfile(src_path, dst_path)
                except Exception as e:
                    print(f"Skipped: {file} → {e}")
    print("Dataset sorting complete.")


def split_dataset(source_dir, test_dest_dir, ratio=0.8):
    for category in ['healthy', 'diseased']:
        src = os.path.join(source_dir, category)
        dst = os.path.join(test_dest_dir, category)
        os.makedirs(dst, exist_ok=True)
        files = [f for f in os.listdir(src) if f.endswith(('.jpg', '.png'))]
        random.shuffle(files)
        for f in files[int(len(files)*ratio):]:
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    print(" Dataset split into train/test.")

class LeafDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(LeafDiseaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def train_model(train_loader, model, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f" Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(project_dir, "leaf_disease_model.pth"))
    print("✅ Model saved.")

def test_model(test_loader, model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    print(f" Test Accuracy: {100 * correct / total:.2f}%")

def classify_unknown(model, input_folder, output_folder):
    os.makedirs(os.path.join(output_folder, "healthy"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "diseased"), exist_ok=True)
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    model.eval()
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(os.path.join(input_folder, filename)).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                _, pred = torch.max(model(tensor), 1)
            label = "healthy" if pred.item() == 0 else "diseased"
            shutil.copy(os.path.join(input_folder, filename),
                        os.path.join(output_folder, label, filename))
            print(f"{filename} → {label}")
    print("Unknown images sorted.")

if __name__ == "__main__":
    sort_dataset(raw_dataset, os.path.join(project_dir, "dataset"))
    split_dataset(train_dir, test_dir, ratio=0.8)

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    full_train = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    def cap_dataset(dataset, max_per_class=500):
        indices = {i: [] for i in range(len(dataset.classes))}
        for i, (_, label) in enumerate(dataset):
            if len(indices[label]) < max_per_class:
                indices[label].append(i)
        return Subset(dataset, [i for sub in indices.values() for i in sub])

    train_loader = DataLoader(cap_dataset(full_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    cnn = LeafDiseaseCNN(num_classes=2).to(device)
    train_model(train_loader, cnn)
    test_model(test_loader, cnn)

if os.path.exists(unknown_dir):
    
    classify_unknown(cnn, unknown_dir, sorted_dir)
else:   
    print(f"Skipped classification. Folder not found: {unknown_dir}")
