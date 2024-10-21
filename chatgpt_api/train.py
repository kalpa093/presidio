import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.metrics import precision_recall_fscore_support

# JSON 파일 경로
json_file_path = 'converted_tagged_sentences.json'

# JSON 파일 읽기
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터 확인
print(f"First item: {data[0]}")

# 텍스트와 레이블 추출
texts = [sentence_data['sentence'] for batch in data for sentence_data in batch]
labels = [sentence_data['labels'] for batch in data for sentence_data in batch]


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        labels = self.labels[item]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        # Padding labels
        encoded_labels = labels + [0] * (self.max_len - len(labels))
        encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoded_labels
        }


# BERT 모델과 토크나이저
MODEL_NAME = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=23)

# 데이터셋 및 데이터로더
MAX_LEN = 128
BATCH_SIZE = 8

dataset = CustomDataset(texts, labels, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 훈련 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()


# 학습 함수
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model = model.train()
    losses = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(losses) / len(losses)


# 평가 함수
def evaluate(model, dataloader, device):
    model = model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            true_labels.extend(labels.cpu().numpy().flatten())
            pred_labels.extend(predictions.cpu().numpy().flatten())

    # 각 레이블에 대한 precision, recall, f1-score 계산
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, labels=list(range(23)),
                                                                     zero_division=0)

    # 레이블 0의 가중치를 낮추기 위해 가중치를 수동으로 설정
    custom_weights = support.copy()
    custom_weights[0] *= 0  # 레이블 0의 가중치를 낮춤

    # 가중 평균 계산
    weighted_precision = (precision * custom_weights).sum() / custom_weights.sum()
    weighted_recall = (recall * custom_weights).sum() / custom_weights.sum()
    weighted_f1 = (f1 * custom_weights).sum() / custom_weights.sum()

    return precision, recall, f1, support, weighted_precision, weighted_recall, weighted_f1


# 학습 및 평가 루프
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, dataloader, loss_fn, optimizer, device)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Training Loss: {train_loss}')

# 평가
precision, recall, f1, support, weighted_precision, weighted_recall, weighted_f1 = evaluate(model, dataloader, device)

# 각 레이블에 대한 결과 출력
for label in range(23):
    print(
        f"Label {label} - Precision: {precision[label]}, Recall: {recall[label]}, F1-Score: {f1[label]}, Support: {support[label]}")

# 가중 평균 결과 출력
print(f"Weighted Precision: {weighted_precision}, Weighted Recall: {weighted_recall}, Weighted F1-Score: {weighted_f1}")
