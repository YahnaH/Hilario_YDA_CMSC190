from utils import dct_feat, dwt_feat, concat_csv
import os
from PIL import Image as img

TARGET_SIZE = 200

#called by app to get features from images
#calls on dct_feat and dwt_feat
def get_features(img_path, params): 
    
    print(params)
    
    if params == 0: ##dct
        csvf = 'img.csv'   
        dct_feat.process_img(img_path, csvf, False)
    elif params == 1:  #dwt
         csvf = 'img.csv'
         dwt_feat.process_img(img_path, csvf, False)
    
    else: ##both features
        csvf_0 = 'img-dct.csv'
        csvf_1 = 'img-dwt.csv'
        
        dct_feat.process_img(img_path, csvf_0, False)
        dwt_feat.process_img(img_path, csvf_1, False)
        
        #concat csv files
        dct = 'img-dct.csv'
        dwt = 'img-dwt.csv'
        out = 'img.csv'

        col_ignore = ['entropy','target']

        concat_csv.concat_csv(dct,dwt,out,col_ignore)
        
        os.remove(dct)
        os.remove(dwt)
    
    return


