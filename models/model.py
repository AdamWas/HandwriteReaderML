from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def load_model_and_processor(pretrained_model_name="microsoft/trocr-large-handwritten"):
    # Załaduj procesor i model
    processor = TrOCRProcessor.from_pretrained(pretrained_model_name)
    model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name)

    # Konfiguracja modelu
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = 512  # Obsługa dłuższych tekstów
    processor.tokenizer.model_max_length = 512

    return model, processor
