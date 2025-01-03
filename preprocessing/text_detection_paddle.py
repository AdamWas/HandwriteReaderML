from paddleocr import PaddleOCR

def detect_text_regions(image_path):
    """
    Wykrywa regiony tekstowe na obrazie i rozpoznaje ich zawartość.
    """
    # Inicjalizacja PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en")  # Możesz dostosować język

    # Wykrywanie i rozpoznawanie tekstu
    result = ocr.ocr(image_path, cls=True)

    # Lista rozpoznanych słów
    recognized_words = [line[1][0] for line in result[0]]

    return recognized_words
