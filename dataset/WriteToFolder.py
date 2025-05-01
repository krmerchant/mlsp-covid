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
import csv


DIST = [10]
ANGL = [np.pi/2]
distIdx = [3]
anglIdx = [2]

for iii in range(len(DIST)):
    for jjj in range(len(ANGL)):
        # Specify the distance and angle as 1d arrays
        dist = np.array([DIST[iii]])
        angl = np.array([ANGL[jjj]])
        folderName = f'full_dist_{distIdx[iii]}_angl_{anglIdx[jjj]}'

        '''
        if norm:
            folderName = 'n_' + folderName
        '''
        norm = True

        track_dist_angl = [folderName, dist, angl]
        #track_filename = "test_Dist_Angl.csv"
        track_filename = "covid_dataset.csv"
        # Append the new row to the CSV file
        with open(track_filename, mode="a", newline="") as file:  # Open in append mode
            writer = csv.writer(file)
            writer.writerow(track_dist_angl)

        # Define the path where you want to create the folder
        folder_path = Path(folderName)  # for specific distance and angle

        # Create the folder
        folder_path.mkdir(parents=True, exist_ok=True)

        print(f"Folder '{folder_path}' created successfully.")

        # Call LungDataset and specify distance and angle
        #dataset = LungDataset(csv_file=r'C:\Users\Dory\Documents\GitHub\mlsp-covid\dataset\csv\covid_dataset.csv', root_dir=r'C:\Users\Dory\Documents\JHU\Machine Learning for Signal Processing\COVID-19_Radiography_Dataset')    
        dataset = LungDataset(csv_file=r'C:\Users\Dory\Documents\GitHub\mlsp-covid\dataset\csv\test_file.csv', root_dir=r'C:\Users\Dory\Documents\JHU\Machine Learning for Signal Processing\COVID-19_Radiography_Dataset')    

        start_time = time.time()  # Start the timer

        # Extract GLCM features for each image in the dataset and store in a dictionary json file
        for i in range(len(dataset)):
        #for j in range(300):
        #    for k in [0, 12386, 14462]:
        #        i = j+k
            
            image, lung_mask, label1, label2, masked_image, filename = dataset[i]

            features, glcm = extract_GLCM_features(masked_image, dist, angl, norm)

            image_features = {"contrast": features[0], "energy": features[1], "entropy": features[2],
                            "correlation": features[3], "dissimilarity": features[4], "homogeneity": features[5],
                            "ASM": features[6], "mean": features[7], "variance": features[8],
                            "std": features[9]}  

            image_features["datasetID"] = i
            image_features["imageName"] = filename
            image_features["label1"] = label1
            image_features["label2"] = label2
            image_features["dist"] = dist.item() #convert 1d array to float for json
            image_features["angl"] = angl.item()
            image_features["normed"] = norm
            #image_features["glcm"] = glcm.tolist()  # Convert NumPy array to list

            # Define the JSON file path inside the folder
            json_file_path = folder_path / f"image_{i}_{folderName}.json"

            # Write dictionary to a file
            with open(json_file_path, "w") as file:
                json.dump(image_features, file, indent=4)  # Use indent=4 for pretty formatting

            print(i)

        end_time = time.time()  # End the timer
        print(f"Script runtime: {end_time - start_time:.2f} seconds")


