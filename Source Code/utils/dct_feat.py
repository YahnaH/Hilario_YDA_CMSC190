#This code takes the images from the dataset
#computes the DCT transformation
#and extract its features


from scipy.fftpack import dct
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import restoration, measure, util
from skimage.feature import blob_dog
import os
import numpy as np
import csv
import pandas as pd

IMG_SIZE = 200 # set fixed length
SLICE_SIZE = 20


np.set_printoptions(threshold=np.inf)

def dct2(a):
    return dct(dct(a.T, norm=None).T, norm=None) 

#pass a dct transformed image
#get the absolute value of the coeffs (magnitude only)
#get the mean for each 20x20 block
#returns a list of features per image
def slice(img):
    slices = []
    img = np.abs(img) 
    
    #get the average per 8x8 block
    for y in range(0, img.shape[0], SLICE_SIZE):
        for x in range(0, img.shape[1],SLICE_SIZE):
            block = img[y:y+SLICE_SIZE, x:x+SLICE_SIZE]
            ave = np.mean(block)
            slices.append(ave)
    
    return slices

def compute_mean_image(img):
    im = rgb2gray(imread(img))
    
    imF = dct2(im)
    imF = np.log(np.abs(imF) + 1)
    slices = slice(imF)
    
    #additional features
    entropy = measure.shannon_entropy(im)
    
    slices.append(entropy)
    
    return slices
 
    
#feature extraction
def process_img(dataset_path, csv_file_path, batched):
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
    
    
        feat = int((IMG_SIZE**2) / (SLICE_SIZE**2))
        print(feat)
        feat_list = []      #feature headers
        for i in range(1,feat+1):
            feature = f"feat{i}"
            feat_list.append(feature)
        
        feat_list.append('entropy')
       
        if batched: 
            feat_list.append('target')
        
        csv_writer.writerow(feat_list)
        
        #processing a folder
        if batched: 
            
            subclass_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

            # real/fake
            for subclass_folder in subclass_folders:
                subclass_path = os.path.join(dataset_path, subclass_folder)
                
                image_files = [f for f in os.listdir(subclass_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                for img in image_files:
                    image_path = os.path.join(subclass_path, img)
                    dct_feat = compute_mean_image(image_path)
                    
                    subclass = 1 if subclass_folder == 'fake' else 0
                    
                    #write to csv
                    #calc the mean image per class
                    dct_feat.append(subclass)
                    csv_writer.writerow(dct_feat)
        #single image
        else: 
            features = compute_mean_image(dataset_path)
            csv_writer.writerow(features)
            
    print(f'Values have been saved to {csv_file_path}')
    return


# proces_img(dataset_path, csv_file_path, False)


