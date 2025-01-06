from pipelines.recognition_pipeline import process_image

def main():
    # Ścieżka do obrazu, który chcesz przetworzyć
    image_path = "data/test_image3.jpg"
    
    # Ścieżka do wytrenowanego modelu TrOCR
    model_dir = "models/fine_tuned_trocr"

    print(f"Przetwarzanie obrazu: {image_path}")

    # Wywołanie funkcji przetwarzającej obraz
    try:
        recognized_text = process_image(image_path, model_dir)
        print("Rozpoznany tekst:")
        print(recognized_text)
    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania obrazu: {e}")

if __name__ == "__main__":
    main()
