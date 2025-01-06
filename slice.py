import os
from training.image_crop import detect_text_regions_with_save


def get_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

# Przykład użycia
directory_path = "data/raw_to_crop"
all_files = get_all_files(directory_path)

for file_path in all_files:
    print(file_path)
    detect_text_regions_with_save(str(file_path), output_dir="output/regions_raw")
