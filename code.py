import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from transformers import get_scheduler
from torch.optim import AdamW
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
file_path = "./Threauptic Solution.csv"
df = pd.read_csv(file_path)

# Selecting relevant columns
texts = df['Symptoms'].tolist()
labels = df['Diagnosis / Condition'].tolist()

# Converting labels to numerical values
label_to_id = {label: idx for idx, label in enumerate(set(labels))}
id_to_label = {idx: label for label, idx in label_to_id.items()}
n_classes = len(label_to_id)

labels = [label_to_id[label] for label in labels if label in label_to_id]
print("Unique training labels:", set(labels))
print("Expected label range: 0 to", len(label_to_id) - 1)
assert all(0 <= label < len(label_to_id) for label in labels), "Found an invalid label!"
print("Unique labels in training data:", set(labels))
print("Expected label range: 0 to", len(label_to_id) - 1)

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Apply TF-IDF to convert text into numerical form
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(texts)

# Apply SMOTE for class balancing
from imblearn.over_sampling import SMOTE
ros = SMOTE(random_state=42, k_neighbors=5)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)
X_resampled, labels_resampled = ros.fit_resample(X_tfidf, labels)
train_texts = vectorizer.inverse_transform(X_resampled)
train_texts = [' '.join(words) for words in train_texts]  # Properly convert back to text
train_labels = labels_resampled

# Splitting data
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# Custom Dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Creating datasets
dataset_train = EmotionDataset(train_texts, train_labels, tokenizer)
dataset_val = EmotionDataset(val_texts, val_labels, tokenizer)

# DataLoaders
train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=16)

# Define DistilBERT Classifier Model
class RobertaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RobertaClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT layers
        for param in self.bert.encoder.layer[-4:].parameters():
            param.requires_grad = True  # Unfreeze last 2 layers
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_classes
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        x = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        x = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        x = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.fc(x)
        return logits

# Define Focal Loss with Class Weights
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha if alpha is not None else torch.ones(n_classes)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha.to(inputs.device))
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Training function
def train(model, data_loader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device, dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device, dtype=torch.long).view(-1)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    from sklearn.metrics import accuracy_score, classification_report
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions, target_names=list(label_to_id.keys()))

# Training setup
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaClassifier(n_classes).to(device)
optimizer = AdamW(model.parameters(), lr=5e-6)
total_steps = len(train_dataloader) * 10  # 10 epochs
scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
class_weights = torch.tensor([1.0, 2.0, 2.5, 1.5, 3.0]).to(device)  # Adjust weights based on class distribution
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Training loop
for epoch in range(10):
    print(f"Epoch {epoch + 1}/10")
    train_loss = train(model, train_dataloader, optimizer, scheduler, device, loss_fn)
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)

# Save model
torch.save(model.state_dict(), "distilbert_emotion_classifier.pth")
