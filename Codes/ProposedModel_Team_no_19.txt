# Potato Leaf Disease Detection - VGG16S (PyTorch + CUDA)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score

import numpy as np
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42 
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 2. Dataset and Transforms 
dataset_path = "plantvillage"
img_size = 224
batch_size = 32
SPLIT_RATIO = 0.8

train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) 
])

val_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
train_full_set = datasets.ImageFolder(os.path.join(dataset_path), transform=train_transforms)
val_full_set = datasets.ImageFolder(os.path.join(dataset_path), transform=val_transforms)

train_size = int(SPLIT_RATIO * len(train_full_set))
val_size = len(train_full_set) - train_size
generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(len(train_full_set), generator=generator).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]
train_dataset = Subset(train_full_set, train_indices)
val_dataset = Subset(val_full_set, val_indices) 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
num_classes = len(train_full_set.classes)
class_names = train_full_set.classes
print(f"Total images: {len(train_full_set)}")
print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Classes ({num_classes}): {class_names}")
# 3. VGG16S Model Definition
def vgg16s(pretrained=True, num_classes=3):
    weights = VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)
    
    for param in model.features.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

model = vgg16s(pretrained=True, num_classes=num_classes).to(device)

# 4. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
# Optimizer for Phase 1 
optimizer_phase1 = optim.Adam(model.classifier.parameters(), lr=1e-3)
print("Model initialized and classifier parameters set for Phase 1 optimization.")
# 5. Training - Phase 1 (Freeze Base, Train Classifier)

from tqdm import tqdm # Import TQDM

for param in model.features.parameters():
    param.requires_grad = False

num_epochs_phase1 = 10
train_loss_list, val_loss_list = [], []
phase2_start_epoch = num_epochs_phase1

print(f"\n--- Starting Phase 1: Training Classifier Only for {num_epochs_phase1} Epochs ---")
for epoch in range(num_epochs_phase1):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs_phase1}", unit="step")
    
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer_phase1.zero_grad() 
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer_phase1.step()
        
        running_loss += loss.item() * images.size(0)
        
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        step_loss = loss.item()
        step_acc = (preds == labels).float().mean().item()
        train_bar.set_postfix(loss=f'{step_loss:.4f}', acc=f'{step_acc:.4f}')

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / len(train_loader.dataset)
    train_loss_list.append(epoch_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
    val_loss /= len(val_loader.dataset)
    val_loss_list.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs_phase1} - accuracy: {epoch_accuracy:.4f} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
# 6. Training - Phase 2 (Unfreeze Last Conv Layers - Fine-Tuning)

from tqdm import tqdm 

for i, child in enumerate(model.features.children()):
    if i >= 24: 
        for param in child.parameters():
            param.requires_grad = True

optimizer_phase2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6) 
num_epochs_phase2 = 50

print(f"\n--- Starting Phase 2: Fine-Tuning for {num_epochs_phase2} Epochs (LR=1e-6) ---")
for epoch in range(num_epochs_phase2):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs_phase2}", unit="step")
    
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer_phase2.zero_grad() 
        
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        
        loss.backward()
        optimizer_phase2.step()
        
        running_loss += loss.item() * images.size(0)
        
        
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        
        step_loss = loss.item()
        step_acc = (preds == labels).float().mean().item()
        train_bar.set_postfix(loss=f'{step_loss:.4f}', acc=f'{step_acc:.4f}')

    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / len(train_loader.dataset)
    train_loss_list.append(epoch_loss)

    
    model.eval()
    val_loss = 0.0
    all_preds, all_labels, all_probs = [], [], [] 
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    val_loss /= len(val_loader.dataset)
    val_loss_list.append(val_loss)

    
    print(f"Epoch {epoch+1}/{num_epochs_phase2} - accuracy: {epoch_accuracy:.4f} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
# 7. Metrics 
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)  
num_classes = len(class_names)
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
try:
    auc = roc_auc_score(
        np.eye(num_classes)[all_labels],  
        all_probs,
        multi_class='ovr',
        average='macro'
    )
except ValueError:
    auc = 0.0
specs = []
for cls in range(num_classes):
    tn = np.sum((all_labels != cls) & (all_preds != cls))
    fp = np.sum((all_labels != cls) & (all_preds == cls))
    spec = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    specs.append(spec)
specificity = np.mean(specs)
mcc = matthews_corrcoef(all_labels, all_preds)
kappa = cohen_kappa_score(all_labels, all_preds)
metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "AUC": auc,
    "Specificity": specificity,
    "MCC": mcc,
    "Kappa": kappa
}
print("\nMetrics")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
torch.save(model.state_dict(), "vgg16s_potato_model_final.pth")
print("\nModel saved as vgg16s_potato_model_final.pth")

epochs = range(1, len(train_loss_list)+1)

plt.figure(figsize=(10,6))
plt.plot(epochs, train_loss_list, label="Train Loss", marker='.')
plt.plot(epochs, val_loss_list, label="Validation Loss", marker='.')
plt.axvline(x=phase2_start_epoch, color='red', linestyle='--', label=f"Fine-Tuning Start (Epoch {phase2_start_epoch+1})")

plt.title("Training & Validation Loss (Two-Phase Training)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_final.png')
plt.show()
print("Loss curve saved as loss_curve_final.png")

import matplotlib.ticker as mtick

metrics = {
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1-Score': f1,
    'AUC': auc,
    'Specificity': specificity,
    'MCC': mcc,
    'Kappa': kappa
}

percentage_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC','Specificity','MCC','Kappa']
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.figure(figsize=(10,5))
bars = plt.bar(metric_names, metric_values, color=['skyblue'] * len(percentage_metrics) + ['salmon'])
plt.title('Final Evaluation Metrics')
plt.ylabel('Value')

max_val = max(metric_values)
plt.ylim(0, max(1.05, max_val * 1.1)) 
plt.xticks(rotation=20, ha='right')

for i, name in enumerate(metric_names):
    if name in percentage_metrics and metric_values[i] <= 1:
        plt.text(bars[i].get_x() + bars[i].get_width()/2., metric_values[i] + 0.02, 
                 f'{metric_values[i]:.2%}', ha='center', va='bottom')
            
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('final_metrics_final.png', bbox_inches='tight')
plt.show()
print("Metrics bar plot saved as final_metrics_final.png")

# 8. External Image Validation

from PIL import Image

EXTERNAL_IMAGE_PATH = "healthy.jpg"
model.eval()

external_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

try:
    img = Image.open(EXTERNAL_IMAGE_PATH).convert('RGB')
    
    
    img_tensor = external_transforms(img).unsqueeze(0).to(device) # unsqueeze(0) adds the batch dimension
    
    
    with torch.no_grad():
        output = model(img_tensor)
    
    
    probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index]
    
    
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class} | Confidence: {confidence:.4f}")
    plt.axis('off')
    plt.show()

    print(f"\n--- Prediction Details for {EXTERNAL_IMAGE_PATH} ---")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("\nAll Class Probabilities:")
    
    
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {i}. {name:<25}: {prob:.4f}")

except FileNotFoundError:
    print(f"ERROR: Image file not found at {EXTERNAL_IMAGE_PATH}. Please check the path.")
except Exception as e:
    print(f"An error occurred during prediction: {e}")
from PIL import Image


EXTERNAL_IMAGE_PATH = "early.jpg" 
model.eval()

external_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

try:
    
    img = Image.open(EXTERNAL_IMAGE_PATH).convert('RGB')
    
    
    img_tensor = external_transforms(img).unsqueeze(0).to(device) # unsqueeze(0) adds the batch dimension
    
    
    with torch.no_grad():
        output = model(img_tensor)
    
    
    probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index]
    
    
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class} | Confidence: {confidence:.4f}")
    plt.axis('off')
    plt.show()

    print(f"\n--- Prediction Details for {EXTERNAL_IMAGE_PATH} ---")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("\nAll Class Probabilities:")
    
    
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {i}. {name:<25}: {prob:.4f}")

except FileNotFoundError:
    print(f"ERROR: Image file not found at {EXTERNAL_IMAGE_PATH}. Please check the path.")
except Exception as e:
    print(f"An error occurred during prediction: {e}")
from PIL import Image

EXTERNAL_IMAGE_PATH = "late2.webp"
model.eval()

external_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

try:
    
    img = Image.open(EXTERNAL_IMAGE_PATH).convert('RGB')
    
    
    img_tensor = external_transforms(img).unsqueeze(0).to(device) # unsqueeze(0) adds the batch dimension
    
    
    with torch.no_grad():
        output = model(img_tensor)
    
    
    probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index]
    
    
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class} | Confidence: {confidence:.4f}")
    plt.axis('off')
    plt.show()

    print(f"\n--- Prediction Details for {EXTERNAL_IMAGE_PATH} ---")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("\nAll Class Probabilities:")
    
    
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {i}. {name:<25}: {prob:.4f}")

except FileNotFoundError:
    print(f"ERROR: Image file not found at {EXTERNAL_IMAGE_PATH}. Please check the path.")
except Exception as e:
    print(f"An error occurred during prediction: {e}")

