import os
import random
import shutil

# Put all the images in all/images and all the labels in all/labels
# This code will split them on a 80/20 ratio between train and test folder

# Paths
folder_all_images = "C:/Users/blagn771/Desktop/test/all/images"
folder_all_labels = "C:/Users/blagn771/Desktop/test/all/labels"
folder_train_images = "C:/Users/blagn771/Desktop/test/train/images"
folder_train_labels = "C:/Users/blagn771/Desktop/test/train/labels"
folder_test_images = "C:/Users/blagn771/Desktop/test/test/images"
folder_test_labels = "C:/Users/blagn771/Desktop/test/test/labels"

# Clean test and train folders
for folder in [folder_train_images, folder_train_labels, folder_test_images, folder_test_labels]:
	for file in os.listdir(folder):
		file_path = os.path.join(folder, file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(f"Error: {e}" )
            
# List all files
labels = [file for file in os.listdir(folder_all_labels) if file.endswith('.txt')]
images = [file for file in os.listdir(folder_all_images) if (file.endswith('.jpg') or file.endswith('.png'))]

# Calculate 20% of the files
test_size = int(0.2 * len(images))

# Randomly select test files
test_images = random.sample(images, test_size)
train_images = [elmt for elmt in images if elmt not in test_images]

# Copy test labels and images to test folder
for image_file in test_images:
	if image_file.replace('png', 'txt') in labels:
		label_file = image_file.replace('png', 'txt')
	elif image_file.replace('jpg', 'txt') in labels:
		label_file = image_file.replace('jpg', 'txt')
	else:
		label_file = 'empty'
	shutil.copy(os.path.join(folder_all_images, image_file), os.path.join(folder_test_images, image_file))
	if label_file in labels:
		shutil.copy(os.path.join(folder_all_labels, label_file), os.path.join(folder_test_labels, label_file))
    	
# Copy the rest of the files to train folder
for image_file in train_images:
	if image_file.replace('png', 'txt') in labels:
		label_file = image_file.replace('png', 'txt')
	elif image_file.replace('jpg', 'txt') in labels:
		label_file = image_file.replace('jpg', 'txt')
	else:
		label_file = 'empty'
	shutil.copy(os.path.join(folder_all_images, image_file), os.path.join(folder_train_images, image_file))
	if label_file in labels:
		shutil.copy(os.path.join(folder_all_labels, label_file), os.path.join(folder_train_labels, label_file))
    	
