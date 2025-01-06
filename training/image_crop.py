from pathlib import Path
from paddleocr import PaddleOCR
from PIL import Image
import os

def detect_text_regions_with_save(image_path, output_dir="output/regions", lang="pl"):
    """
    Wykrywa regiony tekstowe za pomocą PaddleOCR i zapisuje wycięte fragmenty do plików.
    
    Args:
        image_path (str): Ścieżka do obrazu wejściowego.
        output_dir (str): Katalog, w którym zapisywane będą fragmenty.
        lang (str): Język OCR (domyślnie "en").
        
    Returns:
        list: Lista rozpoznanych słów.
    """
    # Inicjalizacja PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    # Detekcja tekstu
    result = ocr.ocr(image_path, cls=True)

    # Utwórz katalog na fragmenty, jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)

    # Wczytaj obraz
    image = Image.open(image_path)

    recognized_words = []

    # Iteruj przez wykryte regiony
    for idx, line in enumerate(result[0]):
        box = line[0]  # Współrzędne regionu
        word = line[1][0]  # Rozpoznany tekst
        recognized_words.append(word)

        # Wytnij fragment obrazu
        x_min, y_min = min(box, key=lambda x: x[0])[0], min(box, key=lambda x: x[1])[1]
        x_max, y_max = max(box, key=lambda x: x[0])[0], max(box, key=lambda x: x[1])[1]
        cropped_image = image.crop((x_min, y_min, x_max, y_max))

        filename = Path(image_path).name
        # Zapisz fragment jako obraz
        cropped_image.save(os.path.join(output_dir, f"{filename}_{idx+1}.png"))

    return recognized_words
