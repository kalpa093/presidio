import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=16):
        self.data = pd.read_csv(csv_file, encoding='cp949')
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.labels = self.data['label'].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        label_idx = torch.tensor(self.labels.tolist().index(label))

        combined_text = f"{text} "

        encoding = self.tokenizer(combined_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask, label_idx

def evaluate_model(model, dataloader, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_samples += len(labels)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / num_samples

    return accuracy, precision, recall, f1, avg_loss

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

csv_file_path = "train_name.csv"
dataset = CustomDataset(csv_file=csv_file_path, tokenizer=tokenizer)
batch_size = 16
shuffle = True
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_data, val_data = train_test_split(dataset, test_size=0.4, random_state=93)
val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=93)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(dataset.labels))

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model.to(device)

criterions = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-7, no_deprecation_warning=True)

train_accuracy_history = []
train_f1_history = []
val_accuracy_history = []
val_f1_history = []

num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    acc, pre, rec, f, val_loss = evaluate_model(model, val_dataloader, criterions)
    train_accuracy, train_f1, _1, _2, _3 = evaluate_model(model, train_dataloader, criterions)
    train_accuracy_history.append(train_accuracy)
    train_f1_history.append(train_f1)
    val_accuracy_history.append(acc)
    val_f1_history.append(f)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    print(f"Validation Accuracy: {acc:.4f}, Validation Precision: {pre:.4f}, Validation Recall: {rec:.4f}, Validation F1: {f:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_accuracy_history, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_f1_history, label='Training F1 Score')
plt.plot(range(1, num_epochs + 1), val_f1_history, label='Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Training and Validation F1 Score')

plt.tight_layout()
plt.show()
