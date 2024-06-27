# Identification of GAN-Generated Images through Frequency Analysis and Machine Learning

## Author: Yanna Denise A. Hilario
### Adviser: Asst. Prof. Rodolfo C. Camaclang III
### AY: 2023-2024

### Abstract
The technological advancement of Generative Adversarial Networks (GANs) has allowed the creation of synthetic images, posing a threat for digital disinformation and media fabrication. Due to this, numerous methods have been proposed to counter GAN-generated media. However, most methods employ deep learning and the use of Convolutional Neural Networks (CNNs), which can be computationally expensive to train. This study proposes frequency analysis through Discrete Cosine Transform (DCT) and Discrete Wavelet Transform (DWT) as distinct pre-processing methods for GAN image classification. Additionally, this study uses a Support Vector Machine (SVM) model to classify fake from real images. To address the limitations of using faces as the primary object class, this study investigates the generalizability of GAN traces in the frequency domain across various object classes using the ProGAN dataset. The study found that using DCT as a pre-processing method provides the most significant performance among the proposed methods, with an accuracy of 97.08\%.


#### To run:

pip install -r requirements.txt
pip app.py

#### App Description:
This application is used to identify real from GAN-generated images using Discrete Cosine Transform (DCT) and Discrete Wavelet Transform (DWT). 

##### Required files:
-test-both.csv && train-both.csv
-test-dct.csv  && train-dct.csv
-test-dwt.csv  && train-dwt.csv

#### To use:
Make sure you have the images stored inside a directory.
Select the image from the directory and select the feature extraction method.
(See demo video for reference)
