from datasets import LungDataset
from matplotlib import pyplot as plt
import torch
from skimage import io
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

#this is for the extracted features dataset

class GLCM_Features_LungDataset(LungDataset):
    def __init__ (self, dataset, distances, angles):
        self.orig_dataset = dataset
        self.distances = distances
        self.angles = angles

        all_data_features = []
        for masked_image in self.orig_dataset:
            GLCM = graycomatrix(masked_image, distances, angles)
            prop_contrast = graycoprops(GLCM, 'contrast')
            prop_energy = graycoprops(GLCM, 'energy')
            prop_entropy = graycoprops(GLCM, 'entropy')
            prop_correlation = graycoprops(GLCM, 'correlation')
            prop_dissimilarity = graycoprops(GLCM, 'dissimilarity')
            prop_homogeneity = graycoprops(GLCM, 'homogeneity')

            features = {'contrast': prop_contrast,
                        'energy': prop_energy,
                        'entropy': prop_entropy,
                        'correlation': prop_correlation,
                        'dissimilarity': prop_dissimilarity,
                        'homogeneity': prop_homogeneity,
                        'filename': dataset.dataset_csv,
                         }
            all_data_features.append(features)
        return all_data_features
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return_image = torch.tensor(io.imread(image_name))
            

