# nishad_2022_fast.py
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
def evaluate_metrics(y_true, y_pred, y_prob, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specs=[]
    for i in range(num_classes):
        tn = cm.sum() - (cm[i,i]+cm[i,:].sum()-cm[i,i]+cm[:,i].sum()-cm[i,i])
        fp = cm[:,i].sum()-cm[i,i]
        specs.append(tn/(tn+fp) if tn+fp>0 else 0)
    return {
        "Accuracy": accuracy_score(y_true,y_pred),
        "Precision": precision_score(y_true,y_pred,average='macro',zero_division=0),
        "Recall": recall_score(y_true,y_pred,average='macro',zero_division=0),
        "F1": f1_score(y_true,y_pred,average='macro',zero_division=0),
        "AUC": roc_auc_score(y_true,y_prob,multi_class='ovr'),
        "Specificity": np.mean(specs),
        "MCC": matthews_corrcoef(y_true,y_pred),
        "Kappa": cohen_kappa_score(y_true,y_pred)
    }
def kmeans_mask(img):
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    Z = img_bgr.reshape((-1,3)).astype(np.float32)
    _, labels, _ = cv2.kmeans(Z,2,None,
                              (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1),
                              3,cv2.KMEANS_RANDOM_CENTERS)
    mask = (labels.flatten()==labels.mean()).astype(np.uint8).reshape(img_bgr.shape[:2])
    return Image.fromarray(np.array(img)*mask[:,:,None])

class KMeansDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path).convert("RGB")
        img = kmeans_mask(img)   
        if self.transform: img = self.transform(img)
        return img, target

data_dir = "plantvillage"
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
full_dataset = KMeansDataset(data_dir, transform=transform)
train_size = int(0.8*len(full_dataset))
test_size = len(full_dataset) - train_size
train_ds, test_ds = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False   
model.classifier[6] = nn.Linear(4096, len(full_dataset.classes))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
epochs = 50   
for ep in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    print(f"Epoch {ep+1}/{epochs} - Accuracy: {correct/total:.4f} - Loss: {total_loss/total:.4f}")
y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for x, y in test_loader:
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
