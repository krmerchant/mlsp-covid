from pathlib import Path
import numpy as np
import json
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from extract_GLCM_features import extract_GLCM_features
from datasets import LungDataset
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import time
import re

superkeys = {
    "contrast",
    "energy",
    "entropy",
    "correlation",
    "dissimilarity",
    "homogeneity",
    "ASM",
    "mean",
    "variance",
    "std",
    "datasetID",
    "imageName",
    "label1",
    "label2",
    "dist",
    "angl"
}
removedkeys = {
    "ASM",
    "mean",
    "variance",
    "std",
    "datasetID",
    "imageName",
    "label1",
    "label2",
    "dist",
    "angl"
}

#orderCombo = ['dist_1_angl_1', 'dist_1_angl_2', 'dist_2_angl_1']
orderCombo = ['dist_1_angl_1']

newkeys = list(superkeys - removedkeys) #now a list
num_features = len(newkeys)

#Initialize the feature matrix and label matrix
#will just go to one folder to get length of json files, assuming the same for each folder
folder_path = Path(orderCombo[0]) #for specific distance and angle 
filenames = list(folder_path.glob("*.json")) #list of all json files in the folder
num_json_files = len(filenames)

feature_data = np.zeros((num_json_files, num_features*len(orderCombo)))
label1_data = np.zeros((num_json_files, 1))
label2_data = np.zeros((num_json_files, 1))

for k, combo in enumerate(orderCombo):

    folder_path = Path(combo) #for specific distance and angle 
    filenames = list(folder_path.glob("*.json")) #list of all json files in the folder
    filenames = sorted(filenames, key=lambda x: int(re.search(r'\d+', x.name).group())) #sort the filenames based on the number in the filename
    num_json_files = len(filenames)

    start_time = time.time()  # Start the timer

    for i in range(num_json_files):
        json_file_path = folder_path / filenames[i].name #path of the json file

        # Read dictionary from the file
        with open(json_file_path, "r") as file:
            loaded_data = json.load(file) #dictionary

            label1_data[i] = int(loaded_data["label1"])
            label2_data[i] = int(loaded_data["label2"])

            for j in range(num_features):
                feature_data[i, (j+(k*num_features))] = loaded_data[newkeys[j]] #list of desired features


label1_data = label1_data.flatten()
label2_data = label2_data.flatten()  

end_time = time.time()  # End the timer
print(f"Script runtime: {end_time - start_time:.2f} seconds")
print(newkeys)
print(label1_data)
print(feature_data.shape)
#for f in filenames:
#    print(f.name)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score

#X_train, X_test, y_train, y_test = train_test_split(feature_data, label1_data, test_size=0.3, random_state=42)

'''
# Create an instance of the Support Vector Classifier (SVC) with a linear kernel
model = SVC(kernel='linear', class_weight='balanced') # Use class_weight='balanced' to handle class imbalance

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print accuracy score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))  # Print detailed classification report
'''

glcm = graycomatrix(masked_image, [5], [0], normed = 'False')