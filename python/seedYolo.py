import os
import random
import shutil

# Paths
folder_label_raw = "C:/Users/blagn771/Desktop/nacaDataset/rawLabel"
folder_image_raw = "C:/Users/blagn771/Desktop/nacaDataset/raw"
folder_label_train = "C:/Users/blagn771/Desktop/nacaDataset/train/labels"
folder_image_train = "C:/Users/blagn771/Desktop/nacaDataset/train/images"
folder_label_test = "C:/Users/blagn771/Desktop/nacaDataset/test/labels"
folder_image_test = "C:/Users/blagn771/Desktop/nacaDataset/test/images"

# Clean test and train folders
for folder in [folder_label_test, folder_image_test, folder_label_train, folder_image_train]:
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error: {e}")

# List all raw files
label_files = [file for file in os.listdir(folder_label_raw) if file.endswith('.txt')]
image_files = [file for file in os.listdir(folder_image_raw) if file.endswith('.png')]

# Calculate 20% of the files
test_size = int(0.2 * len(label_files))

# Randomly select test files
test_labels = random.sample(label_files, test_size)

# Copy test labels and images to test folder
for label_file in test_labels:
    image_file = label_file.replace('.txt', '.png')
    if image_file in image_files:
        shutil.copy(os.path.join(folder_label_raw, label_file), os.path.join(folder_label_test, label_file))
        shutil.copy(os.path.join(folder_image_raw, image_file), os.path.join(folder_image_test, image_file))

# Copy the rest of the files to train folders
for label_file in label_files:
    image_file = label_file.replace('.txt', '.png')
    if image_file in image_files:
        shutil.copy(os.path.join(folder_label_raw, label_file), os.path.join(folder_label_train, label_file))
        shutil.copy(os.path.join(folder_image_raw, image_file), os.path.join(folder_image_train, image_file))
