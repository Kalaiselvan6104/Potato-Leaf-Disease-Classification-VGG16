# shah_2025_pldc_net.py
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix


def evaluate_metrics(y_true, y_pred, y_prob, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specs=[]
    for i in range(num_classes):
        tn = cm.sum() - (cm[i,i]+cm[i,:].sum()-cm[i,i]+cm[:,i].sum()-cm[i,i])
        fp = cm[:,i].sum()-cm[i,i]
        specs.append(tn/(tn+fp) if tn+fp>0 else 0)
    results = {
        "Accuracy": accuracy_score(y_true,y_pred),
        "Precision": precision_score(y_true,y_pred,average='macro',zero_division=0),
        "Recall": recall_score(y_true,y_pred,average='macro',zero_division=0),
        "F1": f1_score(y_true,y_pred,average='macro',zero_division=0),
        "AUC": roc_auc_score(y_true,y_prob,multi_class='ovr'),
        "Specificity": np.mean(specs),
        "MCC": matthews_corrcoef(y_true,y_pred),
        "Kappa": cohen_kappa_score(y_true,y_pred)
    }
    return results
data_dir = "plantvillage"
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.8*len(full_dataset))
test_size = len(full_dataset)-train_size
train_ds, test_ds = random_split(full_dataset,[train_size,test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PLDCNet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b1", pretrained=True, num_classes=0, global_pool="avg")
        self.fc = nn.Sequential(
            nn.Linear(self.backbone.num_features,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        return self.fc(self.backbone(x))

model = PLDCNet(len(full_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
epochs = 50

for ep in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x,y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    train_loss = total_loss/total
    train_acc = correct/total
    print(f"Epoch {ep+1}/{epochs} - Accuracy: {train_acc:.4f} - Loss: {train_loss:.4f}")
model.eval()
y_true=[]
y_pred=[]
y_prob=[]
with torch.no_grad():
    for x,y in test_loader:
        x = x.to(device)
        out = model(x)
        prob = torch.softmax(out,1).cpu().numpy()
        pred = prob.argmax(1)
        y_true.extend(y.numpy())
        y_pred.extend(pred)
        y_prob.extend(prob)
y_prob = np.array(y_prob)
results = evaluate_metrics(y_true, y_pred, y_prob, len(full_dataset.classes))
print("\nTest Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
