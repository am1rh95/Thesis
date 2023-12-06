import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
language = 'de'

train_data = load_dataset('xnli', split='train',language=language)
valid_data = load_dataset('xnli', split='validation',language=language)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




# Load XLM-RoBERTa tokenizer and model
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Assuming three labels (entailment, neutral, contradiction)

# Move model to GPU
model.to(device)

def tokenize_batch(batch):
    return tokenizer(batch["premise"], batch["hypothesis"], padding=True, truncation=True)

tr_data = train_data.map(tokenize_batch, batched=True)
val_data = valid_data.map(tokenize_batch, batched=True)

tr_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./xlmroberta-xnli",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="no"
    
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tr_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()
model.save_pretrained("./xlmroberta-xnli-de.pt")