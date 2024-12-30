import pandas as pd
from datasets import Dataset
from PIL import Image
import torch
import os


def load_dataset(csv_path, images_folder="./data/images"):
    data = pd.read_csv(csv_path, sep=";")  # Separator `;`

    # Dodaj pełne ścieżki do plików obrazów z rozszerzeniem .jpg
    data["image_path"] = data["guid"].apply(
        lambda x: os.path.join(images_folder, f"{x}.jpg")
    )

    # Filtruj tylko istniejące pliki
    data = data[data["image_path"].apply(os.path.exists)].reset_index(drop=True)

    return Dataset.from_pandas(data)


def preprocess_data(example, processor):
    image = Image.open(example["image_path"]).convert("RGB")
    example["image"] = image
    return example


def preprocess_for_training(example, processor):
    # Przetwórz obraz
    pixel_values = processor.feature_extractor(
        images=example["image"], return_tensors="pt"
    ).pixel_values
    # Tokenizuj tekst
    labels = processor.tokenizer(
        example["text"], return_tensors="pt", padding="max_length", truncation=True
    ).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    example["pixel_values"] = pixel_values.squeeze()
    example["labels"] = labels.squeeze()
    return example


def create_dataloader(dataset, processor, batch_size=4):
    # Mapujemy dane
    dataset = dataset.map(lambda x: preprocess_data(x, processor))
    dataset = dataset.map(lambda x: preprocess_for_training(x, processor), remove_columns=["image", "text", "guid", "image_path"])

    # Funkcja do scalania batchów
    def collate_fn(batch):
        pixel_values = torch.stack([torch.tensor(item["pixel_values"]) for item in batch])
        labels = torch.stack([torch.tensor(item["labels"]) for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

