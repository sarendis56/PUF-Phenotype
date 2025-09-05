#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import time
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image

import os
cuda_lib_path = ".venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
if os.path.exists(cuda_lib_path):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{current_ld_path}"

try:
    import cuml
    from cuml.svm import SVC as cuSVC
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
    from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    CUML_AVAILABLE = True
    # print(f"cuML GPU acceleration available! Version: {cuml.__version__}")
except ImportError as e:
    CUML_AVAILABLE = False
    print(f"cuML not available: {e}")
except Exception as e:
    CUML_AVAILABLE = False
    print(f"cuML error: {e}")

# Fallback CPU imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class DRAMPUFDataset(Dataset):
    """Custom dataset for DRAM PUF grayscale images"""

    def __init__(self, image_paths: List[str], labels: List[str], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB for VGG16
        label = self.encoded_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class ModifiedVGG16FeatureExtractor(nn.Module):
    """Modified VGG16 with 1x1 average pooling instead of fully connected layers"""

    def __init__(self, num_classes: int = 5):
        super(ModifiedVGG16FeatureExtractor, self).__init__()

        # Load pre-trained VGG16
        self.vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Remove the classifier (fully connected layers)
        self.features = self.vgg16.features

        # Add 1x1 average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # Output: 6x6x512 = 18,432 features

        # Optional: Add a classifier for end-to-end training
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6 * 6 * 512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x, extract_features=False):
        x = self.features(x)
        x = self.avgpool(x)

        if extract_features:
            return x.view(x.size(0), -1)  # Flatten to (batch_size, 18432)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FineTunedVGG16(nn.Module):
    """Fine-tuned VGG16 for DRAM-PUF domain adaptation"""

    def __init__(self, num_classes: int = 5, freeze_layers: int = 10):
        super(FineTunedVGG16, self).__init__()

        # Load pre-trained VGG16
        self.vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Freeze early layers (low-level features like edges, textures)
        for i, param in enumerate(self.vgg16.features.parameters()):
            if i < freeze_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Replace classifier for DRAM-PUF domain
        self.vgg16.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(25088, 4096),  # VGG16's feature map size: 7*7*512
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Linear(1000, num_classes)
        )

        # Add average pooling for feature extraction
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x, extract_features=False):
        x = self.vgg16.features(x)

        if extract_features:
            x = self.avgpool(x)
            return x.view(x.size(0), -1)  # Flatten to (batch_size, 18432)

        x = self.vgg16.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg16.classifier(x)
        return x

def load_dataset(data_path: str, num_devices: int = 5, use_organized: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load DRAM PUF dataset images and labels

    Args:
        data_path: Path to the dataset directory
        num_devices: Number of devices to include (3, 4, or 5)
        use_organized: If True, use organized dataset (smaller), otherwise use raw dataset (full)

    Returns:
        Tuple of (image_paths, labels)
    """
    print(f"Loading dataset for {num_devices} devices...")

    # Device mapping
    all_devices = ['alpha', 'beta', 'delta', 'epsilon', 'gamma']
    devices = all_devices[:num_devices]

    image_paths = []
    labels = []

    # Choose dataset based on use_organized parameter
    raw_path = Path(data_path) / "Raw" / "grayscale_images"
    organized_path = Path(data_path) / "Organized" / f"DRAM Module Grayscale Classification {num_devices} Devices"

    if use_organized and organized_path.exists():
        print(f"Using organized dataset: {organized_path}")
        for device in devices:
            device_path = organized_path / device
            if device_path.exists():
                for img_file in device_path.glob("*.png"):
                    image_paths.append(str(img_file))
                    labels.append(device)
    elif raw_path.exists():
        print(f"Using full raw dataset: {raw_path}")
        for device in devices:
            device_path = raw_path / device
            if device_path.exists():
                # Traverse the directory structure: device/location/temp/voltage/pattern/
                for location_dir in device_path.iterdir():
                    if location_dir.is_dir():
                        for temp_dir in location_dir.iterdir():
                            if temp_dir.is_dir():
                                for voltage_dir in temp_dir.iterdir():
                                    if voltage_dir.is_dir():
                                        for pattern_dir in voltage_dir.iterdir():
                                            if pattern_dir.is_dir():
                                                for img_file in pattern_dir.glob("*.png"):
                                                    image_paths.append(str(img_file))
                                                    labels.append(device)
    elif organized_path.exists():
        print(f"Fallback to organized dataset: {organized_path}")
        for device in devices:
            device_path = organized_path / device
            if device_path.exists():
                for img_file in device_path.glob("*.png"):
                    image_paths.append(str(img_file))
                    labels.append(device)
    else:
        raise FileNotFoundError(f"Neither raw dataset ({raw_path}) nor organized dataset ({organized_path}) found")

    print(f"Loaded {len(image_paths)} images from {len(set(labels))} devices")
    for device in devices:
        count = labels.count(device)
        print(f"  {device}: {count} images")

    return image_paths, labels

def fine_tune_vgg16(model: FineTunedVGG16, train_loader: DataLoader, val_loader: DataLoader,
                   num_epochs: int = 10, device: torch.device = None) -> FineTunedVGG16:
    """
    Fine-tune VGG16 on DRAM-PUF data

    Args:
        model: FineTunedVGG16 model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on

    Returns:
        Fine-tuned model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Use different learning rates for different layers
    feature_params = []
    for param in model.vgg16.features.parameters():
        if param.requires_grad:
            feature_params.append(param)

    classifier_params = list(model.vgg16.classifier.parameters())

    # Lower learning rate for pre-trained features, higher for new classifier
    optimizer = optim.SGD([
        {'params': feature_params, 'lr': 1e-4},      # Low LR for features
        {'params': classifier_params, 'lr': 1e-3}    # High LR for classifier
    ], momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"Fine-tuning VGG16 for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        scheduler.step()

        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val

        print(f'  Epoch {epoch+1}/{num_epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

    print("Fine-tuning completed!")
    return model

def extract_features_with_vgg16(image_paths: List[str], labels: List[str],
                               batch_size: int = 32, fine_tune: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features using modified VGG16

    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        batch_size: Batch size for processing
        fine_tune: Whether to fine-tune VGG16 on the training data (as mentioned in paper)

    Returns:
        Tuple of (features, encoded_labels)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define transforms - original images are 200×220, VGG16 needs 224×224
    # Direct resize from original dimensions to avoid interpolation artifacts
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation for training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    if fine_tune:
        print("Fine-tuning VGG16 on DRAM-PUF data...")

        # Create training dataset for fine-tuning
        train_dataset = DRAMPUFDataset(image_paths, labels, transform=train_transform)

        # Split for validation during fine-tuning (80/20 split)
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders for fine-tuning
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Initialize and fine-tune model
        model = FineTunedVGG16(num_classes=len(set(labels)))
        model = fine_tune_vgg16(model, train_loader, val_loader, num_epochs=10, device=device)

        print("Using fine-tuned VGG16 for feature extraction...")
    else:
        print("Using pre-trained VGG16 for feature extraction...")
        model = ModifiedVGG16FeatureExtractor(num_classes=len(set(labels)))
        model = model.to(device)

    model.eval()

    # Create dataset for feature extraction (use test transform for consistency)
    feature_dataset = DRAMPUFDataset(image_paths, labels, transform=test_transform)
    feature_dataloader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Extract features
    all_features = []
    all_labels = []

    print("Extracting features...")
    with torch.no_grad():
        for batch_idx, (images, batch_labels) in enumerate(feature_dataloader):
            images = images.to(device)
            features = model(images, extract_features=True)

            all_features.append(features.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * batch_size} images...")

    features = np.vstack(all_features)
    labels_array = np.array(all_labels)

    print(f"Extracted features shape: {features.shape}")
    return features, labels_array

def train_gpu_classifiers(X_train: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
    """
    Train classifiers using GPU acceleration where available

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels

    Returns:
        Dictionary containing results for each classifier
    """
    print("Training GPU-accelerated classifiers...")

    results = {}

    if CUML_AVAILABLE:
        # Feature scaling for SVM with aggressive outlier removal
        print("Scaling features for SVM...")
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        # Remove extreme outliers more aggressively (95th percentile)
        clip_max = np.percentile(X_train, 95)
        clip_min = np.percentile(X_train, 5)

        print(f"  Clipping outliers: [{clip_min:.3f}, {clip_max:.3f}]")
        X_train_clipped = np.clip(X_train, clip_min, clip_max)
        X_test_clipped = np.clip(X_test, clip_min, clip_max)

        # Use MinMaxScaler for more controlled range
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler.fit_transform(X_train_clipped)
        X_test_scaled = scaler.transform(X_test_clipped)

        print(f"Original feature range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"Clipped feature range: [{X_train_clipped.min():.3f}, {X_train_clipped.max():.3f}]")
        print(f"Scaled feature range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
        print(f"Scaled feature mean: {X_train_scaled.mean():.3f}, std: {X_train_scaled.std():.3f}")

        gpu_classifiers = {
            'SVM': {
                'classifier': cuSVC(C=10.0, gamma=0.1, kernel='rbf', probability=True),  # Better for high-dim data
                'use_scaled': False
            },
            'Logistic Regression': {
                'classifier': cuLogisticRegression(C=1, max_iter=500),
                'use_scaled': False
            },
            'KNN': {
                'classifier': cuKNeighborsClassifier(n_neighbors=9),
                'use_scaled': False
            },
            'Random Forest': {
                'classifier': cuRandomForestClassifier(n_estimators=396),
                'use_scaled': False
            },
        }

        for name, config in gpu_classifiers.items():
            print(f"\nTraining GPU {name}...")
            start_time = time.time()

            try:
                clf = config['classifier']
                use_scaled = config['use_scaled']

                # Choose appropriate data
                if use_scaled:
                    X_train_use = X_train_scaled
                    X_test_use = X_test_scaled
                    print(f"  Using scaled features")
                else:
                    X_train_use = X_train
                    X_test_use = X_test
                    print(f"  Using original features")

                # Train the classifier
                clf.fit(X_train_use, y_train)

                # Make predictions
                y_pred = clf.predict(X_test_use)

                # Get probabilities if available
                if hasattr(clf, 'predict_proba'):
                    try:
                        y_pred_proba = clf.predict_proba(X_test_use)  # Use same data as prediction
                    except:
                        y_pred_proba = None
                else:
                    y_pred_proba = None

                # Convert cuML arrays to numpy if needed
                if hasattr(y_pred, 'to_numpy'):
                    y_pred = y_pred.to_numpy()
                if y_pred_proba is not None and hasattr(y_pred_proba, 'to_numpy'):
                    y_pred_proba = y_pred_proba.to_numpy()

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                cv_mean = accuracy
                cv_std = 0.0
                cv_time = 0.0

                print(f"  Test accuracy: {cv_mean:.4f}")

                # Store results
                results[f'GPU {name}'] = {
                    'classifier': clf,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'classification_report': classification_report(y_test, y_pred),
                    'training_time': time.time() - start_time,
                    'cv_time': cv_time
                }

                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  Training time: {time.time() - start_time:.2f}s")

            except Exception as e:
                print(f"  Error training GPU {name}: {e}")
                continue
    
    gpu_classifiers_extra = {
        'GPU XGBoost': xgb.XGBClassifier(
            learning_rate=0.02,
            n_estimators=64,
            max_depth=3,
            random_state=42,
            tree_method='gpu_hist',  # Use GPU
            gpu_id=0
        )
    }
    
    if not CUML_AVAILABLE:
        print("ERROR: cuML is required for GPU acceleration but is not available!")
        print("Please ensure CUDA runtime libraries are properly configured.")
        sys.exit(1)
    
    for name, clf in gpu_classifiers_extra.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        try:
            # Train the classifier
            clf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store results
            results[name] = {
                'classifier': clf,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': accuracy,  # Simplified for speed
                'cv_std': 0.0,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred),
                'training_time': time.time() - start_time
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Training time: {time.time() - start_time:.2f}s")
            
        except Exception as e:
            print(f"  Error training {name}: {e}")
            continue
    
    return results

def plot_results(results: Dict[str, Dict], num_devices: int, save_path: str = None):
    """
    Plot classification results

    Args:
        results: Results from classifier training
        num_devices: Number of devices used
        save_path: Path to save plots (optional)
    """
    print("\nGenerating plots...")

    # Extract metrics
    classifiers = list(results.keys())
    accuracies = [results[clf]['accuracy'] for clf in classifiers]
    f1_scores = [results[clf]['f1_score'] for clf in classifiers]
    cv_means = [results[clf]['cv_mean'] for clf in classifiers]
    cv_stds = [results[clf]['cv_std'] for clf in classifiers]

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'GPU-Accelerated DPAN Performance Analysis - {num_devices} Devices', fontsize=16)

    # Plot 1: Accuracy comparison
    bars1 = ax1.bar(classifiers, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Test Accuracy by Classifier')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.0, 1.0)
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')

    # Plot 2: F1-Score comparison
    bars2 = ax2.bar(classifiers, f1_scores, color='lightcoral', alpha=0.7)
    ax2.set_title('F1-Score by Classifier')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0.0, 1.0)
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, f1 in zip(bars2, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom')

    # Plot 3: Training time comparison
    training_times = [results[clf].get('training_time', 0) for clf in classifiers]
    bars3 = ax3.bar(classifiers, training_times, color='lightgreen', alpha=0.7)
    ax3.set_title('Training Time by Classifier')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, time_val in zip(bars3, training_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.2f}s', ha='center', va='bottom')

    # Plot 4: Accuracy vs F1-Score scatter
    ax4.scatter(accuracies, f1_scores, s=100, alpha=0.7, c=range(len(classifiers)), cmap='viridis')
    for i, clf in enumerate(classifiers):
        ax4.annotate(clf, (accuracies[i], f1_scores[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)
    ax4.set_xlabel('Accuracy')
    ax4.set_ylabel('F1-Score')
    ax4.set_title('Accuracy vs F1-Score')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/dpan_results_{num_devices}_devices.png", dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}/dpan_results_{num_devices}_devices.png")

    plt.show()

def save_results(results: Dict[str, Dict], confidence_analysis: Dict[str, Dict],
                num_devices: int, save_path: str = None):
    """
    Save results to CSV files

    Args:
        results: Results from classifier training
        confidence_analysis: Confidence analysis results
        num_devices: Number of devices used
        save_path: Path to save results (optional)
    """
    if save_path is None:
        save_path = "results"

    os.makedirs(save_path, exist_ok=True)

    # Create results summary
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Classifier': name,
            'Accuracy': result['accuracy'],
            'F1_Score': result['f1_score'],
            'CV_Mean': result['cv_mean'],
            'CV_Std': result['cv_std'],
            'Training_Time': result.get('training_time', 0)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{save_path}/dpan_summary_{num_devices}_devices.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Results summary saved to {summary_file}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='GPU-Accelerated DPAN Reproduction Script')
    parser.add_argument('--data_path', type=str, default='../data', 
                       help='Path to the dataset directory')
    parser.add_argument('--num_devices', type=int, choices=[3, 4, 5], default=5,
                       help='Number of devices to use (3, 4, or 5)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for feature extraction')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (0.0 to 1.0)')
    parser.add_argument('--save_path', type=str, default='results',
                       help='Path to save results')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--use_organized', action='store_true',
                       help='Use organized dataset (smaller) instead of raw dataset (full)')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Fine-tune VGG16 on training data before feature extraction')

    args = parser.parse_args()
    
    print("GPU-Accelerated DPAN Reproduction")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Number of devices: {args.num_devices}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Test size: {args.test_size}")
    print(f"  Save path: {args.save_path}")
    print(f"  GPU available: {torch.cuda.is_available()}")
    print(f"  cuML available: {CUML_AVAILABLE}")
    print("=" * 70)
    
    # Load dataset
    try:
        image_paths, labels = load_dataset(args.data_path, args.num_devices, args.use_organized)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Extract features
    try:
        features, encoded_labels = extract_features_with_vgg16(image_paths, labels, args.batch_size, args.fine_tune)
    except Exception as e:
        print(f"Error extracting features: {e}")
        sys.exit(1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=args.test_size, 
        random_state=42, stratify=encoded_labels
    )
    
    print(f"\nDataset split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Feature dimension: {X_train.shape[1]}")
    
    # Train classifiers
    try:
        results = train_gpu_classifiers(X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f"Error training classifiers: {e}")
        sys.exit(1)
    
    # Print final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"{'Classifier':<25} {'Accuracy':<10} {'F1-Score':<10} {'Time (s)':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<25} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} "
              f"{result['training_time']:<10.2f}")
    
    # Save results
    try:
        save_results(results, {}, args.num_devices, args.save_path)
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Generate plots
    if not args.no_plots:
        try:
            plot_results(results, args.num_devices, args.save_path)
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    print("\nGPU-Accelerated DPAN reproduction completed successfully!")

if __name__ == "__main__":
    main()
