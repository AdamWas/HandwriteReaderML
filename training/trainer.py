import torch
from transformers import AdamW

def validate_model(model, dataloader, device="cuda"):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss}")
    return average_loss

def train_model(model, train_dataloader, val_dataloader, num_epochs=5, learning_rate=5e-5, device="cuda"):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for batch in train_dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Training Loss: {average_loss}")

        # Walidacja po każdej epoce
        validate_model(model, val_dataloader, device)
