from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

def get_dataloaders(checkpoint, batch_size=8):
    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            truncation=True,
        )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["sentence1", "sentence2", "idx"]
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_dataloader, eval_dataloader
