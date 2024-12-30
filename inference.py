import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image

def predict_text(image_path, model_dir="models/fine_tuned_trocr"):
    # Załaduj model i procesor
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    # Przetwarzanie obrazu
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor.feature_extractor(images=image, return_tensors="pt").pixel_values

    # Generowanie tekstu
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text

if __name__ == "__main__":
    # Ścieżka do nowego obrazu
    image_path = "./data/images/test_image2.jpg"  # Podaj ścieżkę do obrazu
    predicted_text = predict_text(image_path)
    print(f"Predicted Text: {predicted_text}")
