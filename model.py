from transformers import AutoModelForSequenceClassification

def get_model(checkpoint, num_labels=2):
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels
    )
