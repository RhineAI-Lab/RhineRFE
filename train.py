# 加载必要的库
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset


# Config
MODEL_PATH = "./model/Qwen/Qwen2___5-0___5B-Instruct"
DATASET_PATH = "./dataset/sharegpt_gpt4"


# Dataset
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
raw_datasets = load_dataset("ag_news")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)


# Model
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=4)


# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# Train
progress_bar = tqdm(range(num_training_steps))
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# Evaluation
metric = evaluate.load("accuracy")
model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

accuracy = metric.compute()
print(f"Accuracy: {accuracy['accuracy']:.4f}")
