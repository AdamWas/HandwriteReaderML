import torch
from models.model import load_model_and_processor
from preprocessing.data_loader import load_dataset, create_dataloader
from training.trainer import train_model


def main():
    print("Is CUDA enabled?",torch.cuda.is_available())
    
    # Załaduj model i procesor
    model, processor = load_model_and_processor()

    # Załaduj dane
    dataset = load_dataset("./data/labels.csv")

    # Przygotuj DataLoader
    dataloader = create_dataloader(dataset, processor)

    # Trenuj model
    train_model(model, dataloader)

    # Zapisz przetrenowany model
    model.save_pretrained("models/fine_tuned_trocr")
    processor.save_pretrained("models/fine_tuned_trocr")


if __name__ == "__main__":
    main()
