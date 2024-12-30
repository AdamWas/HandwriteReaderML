import pandas as pd
from PIL import Image
import os
import torch

def load_dataset(csv_path, images_folder="./data/images"):
    # Wczytaj dane z pliku CSV
    data = pd.read_csv(csv_path, sep=";")

    # Dodaj ścieżki do obrazów
    data["image_path"] = data["guid"].apply(lambda x: os.path.join(images_folder, f"{x}.jpg"))

    # Filtruj tylko istniejące pliki obrazów
    data = data[data["image_path"].apply(os.path.exists)].reset_index(drop=True)

    return data

def create_dataloader(data, processor, batch_size=2):
    # Funkcja przetwarzająca dane na tensor
    def preprocess_for_training(example):
        # Otwórz obraz i przetwórz na tensory
        image = Image.open(example["image_path"]).convert("RGB")
        pixel_values = processor.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze()

        # Tokenizuj tekst
        labels = processor.tokenizer(
            example["text"], return_tensors="pt", padding="max_length", truncation=True
        ).input_ids.squeeze()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

    # Przetwórz dane w locie
    data = data.apply(preprocess_for_training, axis=1)

    # Funkcja do scalania batchów
    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

    # Utwórz DataLoader
    from torch.utils.data import DataLoader
    return DataLoader(data.to_list(), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
