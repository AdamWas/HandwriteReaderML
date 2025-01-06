from sklearn.model_selection import train_test_split
from models.model import load_model_and_processor
from preprocessing.data_loader import load_dataset, create_dataloader
from training.trainer import train_model


def main():
    # Załaduj model i procesor
    model, processor = load_model_and_processor()

    # Załaduj dane treningowe
    data = load_dataset("data/labels.csv")

    # Podział na trening i walidację
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Przygotowanie DataLoaderów
    train_dataloader = create_dataloader(train_data, processor, batch_size=4)
    val_dataloader = create_dataloader(val_data, processor, batch_size=4)

    # Trenuj model
    train_model(model, train_dataloader, val_dataloader, num_epochs=7)

    # Zapisz wytrenowany model
    model.save_pretrained("models/fine_tuned_trocr")
    processor.save_pretrained("models/fine_tuned_trocr")


if __name__ == "__main__":
    main()
