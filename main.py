from sklearn.model_selection import train_test_split
from models.model import load_model_and_processor
from preprocessing.data_loader import load_dataset, create_dataloader
from training.trainer import train_model, validate_model

def main():
    # Załaduj model i procesor
    model, processor = load_model_and_processor()

    # Załaduj dane
    data = load_dataset("./data/labels.csv")

    # Podziel dane na treningowe i walidacyjne (80% treningowe, 20% walidacyjne)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Przygotuj DataLoader dla obu zbiorów
    train_dataloader = create_dataloader(train_data, processor, batch_size=4)
    val_dataloader = create_dataloader(val_data, processor, batch_size=4)

    # Trenuj model z walidacją
    train_model(model, train_dataloader, val_dataloader)

    # Zapisz przetrenowany model
    model.save_pretrained("models/fine_tuned_trocr")
    processor.save_pretrained("models/fine_tuned_trocr")

if __name__ == "__main__":
    main()
