#Concatenate 2 CSV files for DCT and DWT features

import pandas as pd

def concat_csv(dct,dwt,out,remove_col):
    df_dct = pd.read_csv(dct)
    df_dwt = pd.read_csv(dwt)
    
    #remove some features from DCT file, since used also in DWT
    if remove_col:
        df_dct = df_dct.drop(columns=remove_col,errors='ignore')
    
    
    features = pd.concat([df_dct,df_dwt],axis=1)
    features.to_csv(out,index=False) 
    
    print(f"Concatenated CSV saved as '{out}'")


# dct = '../train-dct.csv'
# dwt = '../train-dwt.csv'
# out = '../train-both.csv'

# col_ignore = ['entropy','target']

# concat_csv(dct,dwt,out,col_ignore)
