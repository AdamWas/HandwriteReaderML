# from transformers import VisionEncoderDecoderModel, TrOCRProcessor
# from PIL import Image
# import torch

# def predict_text(image_path, model_dir="models/fine_tuned_trocr_large_handwritten"):
#     processor = TrOCRProcessor.from_pretrained(model_dir)
#     model = VisionEncoderDecoderModel.from_pretrained(model_dir)

#     # Wczytaj i przetwórz obraz
#     image = Image.open(image_path).convert("RGB")
#     pixel_values = processor.feature_extractor(images=image, return_tensors="pt").pixel_values

#     # Generuj tekst
#     model.eval()
#     with torch.no_grad():
#         generated_ids = model.generate(pixel_values, max_length=512)  # Obsługa dłuższych sekwencji
#         generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     return generated_text

# if __name__ == "__main__":
#     image_path = "data/images/test_image2.jpg"
#     predicted_text = predict_text(image_path)
#     print(f"Predicted Text: {predicted_text}")

# from recognition_pipeline import process_image

# image_path = "data/images/test_image.jpg"
# full_text = process_image(image_path, model_dir="models/fine_tuned_trocr")
# print(f"Recognized Text: {full_text}")





# from paddleocr import PaddleOCR

# ocr = PaddleOCR(use_angle_cls=True, lang="pl")
# result = ocr.ocr("data/images/test_image.jpg", cls=True)

# for line in result[0]:
    # print(f"Text: {line[1][0]}, Confidence: {line[1][1]}")

# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())


from training.image_crop import detect_text_regions_with_save
image_path = "data/images/test_image.jpg"
output_dir = "output/regions"

recognized_words = detect_text_regions_with_save(image_path, output_dir)

print(f"Recognized Words: {recognized_words}")
print(f"Regions saved in: {output_dir}")