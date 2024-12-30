from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import torch

def predict_text(image_path, model_dir="models/fine_tuned_trocr"):
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    # Wczytaj i przetwórz obraz
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor.feature_extractor(images=image, return_tensors="pt").pixel_values

    # Generuj tekst
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=512)  # Obsługa dłuższych sekwencji
        generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

if __name__ == "__main__":
    image_path = "data/images/test_image.jpg"
    predicted_text = predict_text(image_path)
    print(f"Predicted Text: {predicted_text}")