#dwt feature extraction 

import pywt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import restoration, measure, util
from skimage.feature import blob_dog
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

#dwt sizes are half the size of the original image

#level 1 decomposition
def dwt(a, wavelet='haar'):
    coeffs_level1 = pywt.dwt2(a, wavelet)
    LL1, (LH1, HL1, HH1) = coeffs_level1
    
    return  LL1, LH1, HL1, HH1


def compute_mean_image(img):
    im = rgb2gray(imread(img))
    
    entropy = measure.shannon_entropy(im)
    
    #image decomposition using DWT    
    LL, LH, HL, HH = dwt(im)
    
    LL_entropy = measure.shannon_entropy(LL)
    LH_entropy = measure.shannon_entropy(LH)
    HL_entropy = measure.shannon_entropy(HL)
    HH_entropy = measure.shannon_entropy(HH)
    
    LL_float = util.img_as_float(LL)
    LH_float = util.img_as_float(LH)
    HL_float = util.img_as_float(HL)
    HH_float = util.img_as_float(HH)
    
    
    LL_var = np.var(LL_float)
    LL_std_dev = np.std(LL_float)
    
    LH_var = np.var(LH_float)
    LH_std_dev = np.std(LH_float)
    
    HL_var = np.var(HL_float)
    HL_std_dev = np.std(HL_float)
    
    HH_var = np.var(HH_float)
    HH_std_dev = np.std(HH_float)
    
    
    LL_blob = len(blob_dog(LL,max_sigma= 8, threshold = 0.1))
    LH_blob = len(blob_dog(LH,max_sigma= 8, threshold = 0.1))
    HL_blob = len(blob_dog(HL,max_sigma= 8, threshold = 0.1))
    
    slice_LL = np.mean((LL))
    slice_LH = np.mean((LH))
    slice_HL = np.mean((HL))
    slice_HH = np.mean((HH))

    slices = []
    
    slices.append(slice_LL)
    slices.append(slice_LH)
    slices.append(slice_HL)
    slices.append(slice_HH)
    
    
    slices.append(LL_var)
    slices.append(LL_std_dev)
    slices.append(LH_var)
    slices.append(LH_std_dev)
    slices.append(HL_var)
    slices.append(HL_std_dev)
    slices.append(HH_var)
    slices.append(HH_std_dev)
    
    slices.append(entropy)
    slices.append(LL_entropy)
    slices.append(LH_entropy)
    slices.append(HL_entropy)
    slices.append(HH_entropy)
    slices.append(LL_blob)
    slices.append(LH_blob)
    slices.append(HL_blob)
    
    
    return slices
    

def process_img(dataset_path, csv_file_path, batched):
    #print(dataset_path, csv_file_path, batched)
    #feature extraction
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        feat_list = []      #feature headers

        feat_list.append('LL mean')
        feat_list.append('LH mean')
        feat_list.append('HL mean')
        feat_list.append('HH mean')
        
        feat_list.append('LL_var')
        feat_list.append('LL_std_dev')
        feat_list.append('LH_var')
        feat_list.append('LH_std_dev')
        feat_list.append('HL_var')
        feat_list.append('HL_std_dev')
        feat_list.append('HH_var')
        feat_list.append('HH_std_dev')
        
        feat_list.append('entropy')
        feat_list.append('LL_entropy')
        feat_list.append('LH_entropy')
        feat_list.append('HL_entropy')
        feat_list.append('HH_entropy')
        feat_list.append('LL_blob')
        feat_list.append('LH_blob')
        feat_list.append('HL_blob')        
        
        if batched: 
            feat_list.append('target')
        
        csv_writer.writerow(feat_list)
    
        if batched: 
            subclass_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

            # real/fake
            for subclass_folder in subclass_folders:
                subclass_path = os.path.join(dataset_path, subclass_folder)
                
                image_files = [f for f in os.listdir(subclass_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                for img in image_files:
                    image_path = os.path.join(subclass_path, img)
                    slices = compute_mean_image(image_path)
                    
                    subclass = 1 if subclass_folder == 'fake' else 0
                    
                    #write to csv
                    #calc the mean image per class
                    slices.append(subclass)
                    csv_writer.writerow(slices)
                    
        #single image
        else: 
            features = compute_mean_image(dataset_path)
            csv_writer.writerow(features)
        
        print(f'Values have been saved to {csv_file_path}')
    return


