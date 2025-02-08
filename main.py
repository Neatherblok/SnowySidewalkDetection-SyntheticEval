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
import argparse  # Add this import at the top

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

def train_on_synthetic_data(epochs=25, batch_size=4, learning_rate=0.001, verbose=False):
    """Train models on synthetic data with configurable parameters."""
    base_data_path, data_folders = download_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_models = {}

    for name, folder_path in data_folders.items():
        if verbose:
            print(f"\n=== Training model on {name} data ===")
        if not os.path.exists(folder_path):
            if verbose:
                print(f"Skipping {name} because folder does not exist: {folder_path}")
            continue

        images, labels = load_images_with_labels(folder_path)
        transform = get_transform()
        dataset = images_to_dataset(images, labels, transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True
        )

        model = CustomModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            loss_val, acc_val = train_model(model, dataloader, criterion, optimizer, device)
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}, Loss={loss_val:.4f}, Acc={acc_val:.2f}%")

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

def load_saved_models(model_dir):
    """Load all saved models from the specified directory."""
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")
    
    for filename in os.listdir(model_dir):
        if filename.endswith("_model.pth"):
            model_name = filename.replace("_model.pth", "")
            model = CustomModel().to(device)
            model.load_state_dict(torch.load(os.path.join(model_dir, filename)))
            model.eval()
            models[model_name] = model
    
    return models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Snow Detection Model Training/Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train new models:
    python main.py train
  Evaluate existing models:
    python main.py eval
  Train and evaluate:
    python main.py train eval
  Evaluate specific models:
    python main.py eval --models model1 model2
  Train with custom parameters:
    python main.py train --epochs 50 --batch-size 8 --learning-rate 0.0001
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train new models')
    train_parser.add_argument('--epochs', type=int, default=25, help='Number of epochs (default: 25)')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Batch size (default: 4)')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    train_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate existing models')
    eval_parser.add_argument('--models', nargs='+', help='Specific models to evaluate (default: all)')
    eval_parser.add_argument('--runs', type=int, default=12, help='Number of evaluation runs (default: 12)')
    eval_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_dir = "./trained_models"
    os.makedirs(model_save_dir, exist_ok=True)
    models = {}

    # Training phase
    if 'train' in args.command:
        if args.verbose:
            print(f"=== Training new models with parameters ===")
            print(f"Epochs: {args.epochs}")
            print(f"Batch size: {args.batch_size}")
            print(f"Learning rate: {args.learning_rate}")
            print(f"Device: {device}")
        
        trained_models = train_on_synthetic_data(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            verbose=args.verbose
        )
        
        # Save the trained models
        for model_name, model in trained_models.items():
            model_path = os.path.join(model_save_dir, f"{model_name}_model.pth")
            torch.save(model.state_dict(), model_path)
            if args.verbose:
                print(f"Saved model to {model_path}")
            models[model_name] = model

    # Evaluation phase
    if 'eval' in args.command:
        if args.verbose:
            print("\n=== Loading pre-trained models for evaluation ===")
            print(f"Number of evaluation runs: {args.runs}")
        
        try:
            loaded_models = load_saved_models(model_save_dir)
            if not loaded_models:
                print("No pre-trained models found in ./trained_models directory")
                if 'train' not in args.command:
                    exit(1)
                    
            # Filter models if specific ones are requested
            if args.models:
                loaded_models = {k: v for k, v in loaded_models.items() if k in args.models}
                if not loaded_models:
                    print(f"None of the specified models were found: {args.models}")
                    exit(1)
                    
            models.update(loaded_models)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            if 'train' not in args.command:
                print("Please ensure you have trained models in the ./trained_models directory")
                exit(1)

    # Evaluate models
    if models:
        for model_name, model in models.items():
            print(f"\n=== Evaluating {model_name} model ===")
            dataloader_func = load_combined_dataset
            f2_scores, confusion_matrices, avg_f2_score = evaluate_model_multiple_runs(
                model, dataloader_func, device, num_runs=args.runs
            )

            print(f"  {model_name} Results:")
            print(f"    F2 Scores: {f2_scores}")
            print(f"    Average F2 Score: {avg_f2_score:.4f}")
            print(f"    Confusion Matrices:")
            for idx, conf_mat in enumerate(confusion_matrices, 1):
                print(f"      Run {idx}:\n{conf_mat}")
                plot_confusion_matrix(conf_mat, f"{model_name} - Run {idx}")

        plt.show()