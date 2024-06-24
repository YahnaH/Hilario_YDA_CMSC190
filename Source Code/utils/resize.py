#utility code for resizing images
#note: resized instead of cropped to preserve details

from PIL import Image
import os

TARGET = 200

def resize_img(folder_path, batched):
    target_size=(TARGET, TARGET)
    if batched: 
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)
                
                img_resized = img.resize(target_size)
            
                img_resized.save(img_path)   
                print(f"{filename} resized to {TARGET} x {TARGET}")
    else:
        img = Image.open(folder_path)
        img_resized = img.resize(target_size)
        
        img_resized.save(folder_path) 
        img_resized.close()  ##close so no error during app usage
        print(f"Image resized to {TARGET} x {TARGET}")
        
        
        return folder_path

