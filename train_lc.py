import torch
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import evaluate
from torch.utils.tensorboard import SummaryWriter

from config import CHECKPOINT, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from data import get_dataloaders
from model import get_model


def main():
    accelerator = Accelerator()
    writer = SummaryWriter("runs/bert-mrpc")

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

    metric = evaluate.load("glue", "mrpc")

    global_step = 0
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(NUM_EPOCHS):

        ########################
        # TRAINING
        ########################
        model.train()
        total_train_loss = 0

        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()

            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            writer.add_scalar("Loss/train_step", loss.item(), global_step)

            global_step += 1
            progress_bar.update(1)

        avg_train_loss = total_train_loss / len(train_dl)

        ########################
        # EVALUATION
        ########################
        model.eval()
        total_eval_loss = 0

        for batch in eval_dl:
            with torch.no_grad():
                outputs = model(**batch)

            total_eval_loss += outputs.loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch["labels"])
            )

            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        avg_eval_loss = total_eval_loss / len(eval_dl)

        ########################
        # LOGGING
        ########################
        accelerator.print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Eval Loss: {avg_eval_loss:.4f} | "
            f"Accuracy: {eval_metric['accuracy']:.4f} | "
            f"F1: {eval_metric['f1']:.4f}"
        )

        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/eval", avg_eval_loss, epoch)
        writer.add_scalar("Metrics/accuracy", eval_metric["accuracy"], epoch)
        writer.add_scalar("Metrics/f1", eval_metric["f1"], epoch)

        metric.reset()

    writer.close()


if __name__ == "__main__":
    main()