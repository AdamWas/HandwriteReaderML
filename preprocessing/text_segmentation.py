from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch


def detect_and_recognize(image_path, model_dir="models/fine_tuned_trocr"):
    """
    Detekcja i rozpoznawanie tekstu na obrazie.
    """
    # Inicjalizacja PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="pl")  # Zmień `lang` na odpowiedni język

    # Detekcja regionów tekstowych
    result = ocr.ocr(image_path, cls=True)

    # Załaduj model rozpoznawania
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    # Lista rozpoznanych słów
    recognized_words = []

    # Rozpoznaj każde wykryte słowo
    for line in result[0]:
        # Wyodrębnij współrzędne i obraz
        box = line[0]  # Współrzędne regionu
        word_image = extract_region(image_path, box)  # Funkcja do wycięcia obrazu

        # Rozpoznawanie tekstu
        recognized_text = recognize_word(word_image, model, processor)
        recognized_words.append(recognized_text)

    return recognized_words


def extract_region(image_path, box):
    """
    Wycięcie fragmentu obrazu na podstawie współrzędnych.
    """
    image = Image.open(image_path)
    x_min, y_min = min(box, key=lambda x: x[0])[0], min(box, key=lambda x: x[1])[1]
    x_max, y_max = max(box, key=lambda x: x[0])[0], max(box, key=lambda x: x[1])[1]
    return image.crop((x_min, y_min, x_max, y_max))


def recognize_word(word_image, model, processor):
    """
    Rozpoznawanie tekstu w wyciętym obrazie słowa.
    """
    pixel_values = processor.feature_extractor(
        images=word_image, return_tensors="pt"
    ).pixel_values
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        recognized_text = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
    return recognized_text
