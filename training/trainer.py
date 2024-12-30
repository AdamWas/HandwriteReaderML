import torch
from transformers import AdamW

def validate_model(model, dataloader, device="cuda"):
    # Walidacja modelu
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()

    # Oblicz średnią stratę walidacyjną
    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss}")
    return average_loss

def train_model(model, train_dataloader, val_dataloader, num_epochs=5, learning_rate=5e-5, device="cuda", accumulation_steps=4):
    # Optymalizator
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Oblicz stratę
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss = loss / accumulation_steps  # Skalowanie straty dla akumulacji gradientów
            total_loss += loss.item()

            # Backpropagation
            loss.backward()

            # Akumulacja gradientów
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

        # Oblicz średnią stratę treningową
        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Training Loss: {average_loss}")

        # Walidacja po każdej epoce
        validate_model(model, val_dataloader, device)
