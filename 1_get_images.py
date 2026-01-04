import pandas as pd
from tqdm import tqdm
import shutil
import os
import numpy as np
import pickle

meta = pd.read_csv('mimic-cxr-2.0.0-metadata.csv')

# PA instances
meta = meta[(meta['ViewPosition'] == 'PA') & 
            (meta['ViewCodeSequence_CodeMeaning'] == 'postero-anterior') &
            (meta['PatientOrientationCodeSequence_CodeMeaning'] == 'Erect')]

# unique studies
print(len(meta)) # 81754
meta = meta.drop_duplicates('study_id')
print(len(meta)) # 72545

# copy images we want to a new directory - unlabelled for now, ignoring
path_to_all_imgs = '../all_mimic_PA_90k/'
all_imgs = os.listdir(path_to_all_imgs)

count = 0
for im in tqdm(all_imgs):
    im_name = im.split('.')[0]
    if im_name in meta['dicom_id'].values:
        shutil.copy(path_to_all_imgs + im, 'filtered_images/' + im)
        count += 1

print(count) # 69895