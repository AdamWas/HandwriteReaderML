from paddleocr import PaddleOCR


def detect_text_regions(image_path):
    """
    Wykrywa regiony tekstowe na obrazie i rozpoznaje ich zawartość.
    """
    # Inicjalizacja PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="pl",  # Możesz dostosować język
        use_gpu=True,
        det_db_box_thresh=0.5,  # Dostosowanie progu wykrywania
        det_db_thresh=0.3,  # Obniżenie progu detekcji
        rec_algorithm="CRNN",  # Algorytm rozpoznawania
    )

    # Wykrywanie i rozpoznawanie tekstu
    result = ocr.ocr(image_path, cls=True)

    # Lista rozpoznanych słów
    recognized_words = [line[1][0] for line in result[0]]

    return recognized_words
