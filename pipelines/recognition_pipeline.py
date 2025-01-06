from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch


def detect_text_regions(image_path):
    """
    Wykrywa regiony tekstowe za pomocą PaddleOCR i zwraca ich współrzędne oraz obrazy.
    """
    # Inicjalizacja PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="pl")  # Zmień `lang` na odpowiedni język

    # Detekcja regionów tekstowych
    result = ocr.ocr(image_path, cls=True)

    # Wyodrębnienie współrzędnych i obrazów
    text_regions = []
    for line in result[0]:
        box = line[0]  # Współrzędne regionu tekstowego
        text_regions.append(box)

    return text_regions


def extract_region(image_path, box):
    image = Image.open(image_path)
    x_min, y_min = min(box, key=lambda x: x[0])[0], min(box, key=lambda x: x[1])[1]
    x_max, y_max = max(box, key=lambda x: x[0])[0], max(box, key=lambda x: x[1])[1]
    cropped_image = image.crop((x_min, y_min, x_max, y_max))

    # Skalowanie do 224x224 (przykładowy rozmiar, dostosuj do modelu)
    cropped_image = cropped_image.resize((224, 224), Image.Resampling.LANCZOS)
    return cropped_image


def recognize_word(word_image, model, processor):
    pixel_values = processor.feature_extractor(
        images=word_image, return_tensors="pt"
    ).pixel_values
    print(f"Pixel values shape: {pixel_values.shape}")  # Logowanie rozmiaru wejścia

    model.eval()
    with torch.no_grad():
        try:
            print("Generowanie id-ków")
            generated_ids = model.generate(
                pixel_values, max_length=50
            )  # Ograniczenie do 50 tokenów)
        except Exception as e:
            print(f"Błąd podczas generowania tekstu: {e}")
            return ""
        recognized_text = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        print("Zwracanie rozpoznanego tekstu")

    return recognized_text


def process_image(image_path, model_dir="models/fine_tuned_trocr"):
    """
    Detekcja regionów tekstowych, wycinanie fragmentów i rozpoznawanie tekstu.
    """
    # Detekcja regionów tekstowych
    print("Detekcja regionów tekstowych...")
    text_regions = detect_text_regions(image_path)
    print("Detekcja zakończona")

    # Załaduj model rozpoznawania tekstu
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    # Rozpoznawanie tekstu w każdym regionie
    recognized_words = []
    for box in text_regions:
        print("Wycięcie regionu")
        word_image = extract_region(image_path, box)
        print("Rozpoznawanie słowa")
        recognized_text = recognize_word(word_image, model, processor)
        recognized_words.append(recognized_text)
        print(f"Rozpoznano: {recognized_text}")

    return " ".join(recognized_words)
