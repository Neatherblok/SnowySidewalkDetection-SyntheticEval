import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics import fbeta_score, confusion_matrix
from collect_images.collect import download_data
from evaluation.confusion_matrices import plot_confusion_matrix
import matplotlib.pyplot as plt

############################
# Example Utility Functions
############################

def load_images_with_labels(folder_path):
    """Load images and labels from subfolders ('snow' and 'clear')."""
    images = []
    labels = []

    for subfolder, label in [("snow", 1), ("clear", 0)]:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.exists(subfolder_path):
            continue
        for file_name in os.listdir(subfolder_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(subfolder_path, file_name)
                img = Image.open(full_path).convert("RGB")
                images.append(img)
                labels.append(label)

    return images, labels

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def images_to_dataset(images, labels, transform=None):
    """Wrap a list of PIL images + labels into a torch Dataset."""
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, lbls, tfms):
            super().__init__()
            self.imgs = imgs
            self.lbls = lbls
            self.tfms = tfms

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            x = self.imgs[idx]
            y = self.lbls[idx]
            if self.tfms:
                x = self.tfms(x)
            return x, y

    return SimpleDataset(images, labels, transform)

#########################
# Example Model Template
#########################

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        pooled = torch.cat((avg_out, max_out), dim=1)
        attn = self.sigmoid(self.conv(pooled))
        return x * attn, attn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.spatial_attention = SpatialAttention(in_channels=64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x, _ = self.spatial_attention(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return epoch_loss, accuracy

##############################
# Train models on each dataset
##############################

def train_on_synthetic_data():
    base_data_path, data_folders = download_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_models = {}

    for name, folder_path in data_folders.items():
        print(f"\n=== Training model on {name} data ===")
        if not os.path.exists(folder_path):
            print(f"Skipping {name} because folder does not exist: {folder_path}")
            continue

        images, labels = load_images_with_labels(folder_path)
        transform = get_transform()
        dataset = images_to_dataset(images, labels, transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        model = CustomModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 25
        for epoch in range(num_epochs):
            loss_val, acc_val = train_model(model, dataloader, criterion, optimizer, device)
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss={loss_val:.4f}, Acc={acc_val:.2f}%")

        trained_models[name] = model

    return trained_models

## Evaluate on the Combined Dataset
def evaluate_model(model, dataloader, device):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability for 'snow' class
            all_outputs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_outputs), np.array(all_labels)

def load_combined_dataset():
    base_data_path, _ = download_data()
    folder = os.path.join(base_data_path, "test")
    
    images = []
    labels = []

    for subdir, label_value in [("snow", 1), ("clear", 0)]:
        sub_path = os.path.join(folder, subdir)
        if not os.path.exists(sub_path):
            print(f'Directory does not exist: {sub_path}')
            continue
        for fn in os.listdir(sub_path):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(sub_path, fn)
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                labels.append(label_value)

    tfm = get_transform()
    dataset = images_to_dataset(images, labels, tfm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    return dataloader

def evaluate_model_multiple_runs(model, dataloader_func, device, num_runs=3):
    f2_scores = []
    confusion_matrices = []

    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")
        dataloader = dataloader_func()
        predictions, true_labels = evaluate_model(model, dataloader, device)
        preds_binary = (predictions > 0.5).astype(int)
        f2 = fbeta_score(true_labels, preds_binary, beta=2)
        conf_mat = confusion_matrix(true_labels, preds_binary)

        f2_scores.append(f2)
        confusion_matrices.append(conf_mat)

        print(f"    F2 score: {f2:.4f}")
        print(f"    Confusion matrix:\n{conf_mat}")

    avg_f2_score = np.mean(f2_scores)
    print(f"  Average F2 score over {num_runs} runs: {avg_f2_score:.4f}")
    return f2_scores, confusion_matrices, avg_f2_score

############################
# Putting It All Together
############################

if __name__ == "__main__":
    # Train and save models
    trained_models = train_on_synthetic_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Folder to save models
    model_save_dir = "./trained_models"
    os.makedirs(model_save_dir, exist_ok=True)

    # Evaluate each model
    for model_name, model in trained_models.items():
        print(f"\n=== Evaluating {model_name} model ===")
        
        # Save the model
        model_path = os.path.join(model_save_dir, f"{model_name}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved to {model_path}")

        # Define dataloader creation function for combined dataset
        dataloader_func = load_combined_dataset

        # Evaluate the model across multiple runs
        f2_scores, confusion_matrices, avg_f2_score = evaluate_model_multiple_runs(
            model, dataloader_func, device, num_runs=12
        )

        # Print summary for this model
        print(f"  {model_name} Results:")
        print(f"    F2 Scores: {f2_scores}")
        print(f"    Average F2 Score: {avg_f2_score:.4f}")
        print(f"    Confusion Matrices:")
        for idx, conf_mat in enumerate(confusion_matrices, 1):
            print(f"      Run {idx}:\n{conf_mat}")
            plot_confusion_matrix(conf_mat, f"{model_name} - Run {idx}")

    plt.show()