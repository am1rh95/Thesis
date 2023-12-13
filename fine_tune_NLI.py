import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, Trainer, TrainingArguments,DataCollatorWithPadding
from datasets import load_dataset
import pandas as pd
language = 'hi'

dataset = load_dataset('xnli',language=language)
# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load XLM-RoBERTa tokenizer and model
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Assuming three labels (entailment, neutral, contradiction)

# Tokenize and preprocess the data
def tokenize_batch(batch):
    return tokenizer(batch["premise"], batch["hypothesis"],  truncation=True)

tokenized_ds  = dataset.map(tokenize_batch, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./xlmroberta-xnli",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="no"
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,

)

# Fine-tune the model
trainer.train()
model.save_pretrained("./xlmroberta-xnli-hi.pt")
tokenizer.save_pretrained("./xlmroberta-xnli-hi.tk")

import json
with open("./xlmroberta-xnli-hi-log-history.json", "w") as f:
    json.dump(trainer.state.log_history, f)