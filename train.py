import torch
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import evaluate

from config import CHECKPOINT, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from data import get_dataloaders
from model import get_model


def main():
    accelerator = Accelerator()

    # Load data and model
    train_dl, eval_dl = get_dataloaders(CHECKPOINT, BATCH_SIZE)
    model = get_model(CHECKPOINT)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model, optimizer, train_dl, eval_dl = accelerator.prepare(
        model, optimizer, train_dl, eval_dl
    )

    num_training_steps = NUM_EPOCHS * len(train_dl)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Load GLUE MRPC metric
    metric = evaluate.load("glue", "mrpc")

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(NUM_EPOCHS):
        ########################
        # TRAINING
        ########################
        model.train()
        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        ########################
        # EVALUATION
        ########################
        model.eval()
        for batch in eval_dl:
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # VERY IMPORTANT for distributed training
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch["labels"])
            )

            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()

        accelerator.print(
            f"Epoch {epoch + 1}: "
            f"Accuracy: {eval_metric['accuracy']:.4f}, "
            f"F1: {eval_metric['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
