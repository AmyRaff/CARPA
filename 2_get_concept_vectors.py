import re
import pandas as pd
import os
from tqdm import tqdm
import pickle
from concept_list_mappings import *
from nlp_utils import *
import collections
#from nlp_experiment import *

metadata = pd.read_csv("mimic-cxr-2.0.0-metadata.csv")
all_images = os.listdir('filtered_images/')

md = metadata.set_index("dicom_id")[["study_id", "subject_id"]]

all_labels = []

id = 0
concept_vectors = []
with open('unfiltered_images.txt', 'w') as imfile, open('unfiltered_image_class_labels.txt', 'w') as g:
    for im in tqdm(all_images):
        im_name = im.rsplit(".", 1)[0]  

        row = md.loc[im_name]
        study = str(row["study_id"])
        subject = str(row["subject_id"])

        report_path = f"../all_mimic_reports/p{subject[:2]}/p{subject}/s{study}.txt"

        with open(report_path, "r") as f:
            report_text = f.readlines()

        cleaned_text = extract_findings_impression(report_text)
        vec = report_to_concept_vector(cleaned_text)
        concept_vectors.append((im, vec))
        
        if sum(vec.values()) > 0:
            labels = get_label_from_vector(vec)
            
            #for lab in labels:
        
            imfile.write(f'{id} {im}\n')
            g.write(f'{id} {labels}\n') # NOTE: multiclass
            id+=1
        
print(id)
pickle.dump(concept_vectors, open('concept_vectors.pkl', 'wb'))