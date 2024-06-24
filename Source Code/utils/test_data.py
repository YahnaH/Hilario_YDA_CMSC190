import os
import random
import shutil

TEST_SIZE = 1    #for real and fake

#utility code to get sample data for test and training

source_folder = 'stylegan2/horse/real'
destination_folder = 'stylegan2-sample/real'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

image_filenames = [filename for filename in os.listdir(source_folder) if filename.endswith(('.jpg', '.png'))]

# Randomly select imgs
selected_image_filenames = random.sample(image_filenames, TEST_SIZE)

for filename in selected_image_filenames:
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)
    shutil.move(source_path, destination_path)

print("Images transferred successfully!")
