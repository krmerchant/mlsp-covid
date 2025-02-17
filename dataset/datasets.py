import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset


class LungDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.root_dir = root_dir
        self.dataset_csv = pd.read_csv(csv_file)

        self.string_to_label = {
            'COVID': 0,
            'NORMAL': 1,
            'VIRAL PNEUMONIA': 2
        }

    def __len__(self):
        return len(self.dataset_csv)

    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        category = self.dataset_csv.iloc[index, 1]
        # pull index from csv file
        image_name = os.path.join(self.root_dir, category,"images/", self.dataset_csv.iloc[index, 0])
        mask_name = os.path.join(self.root_dir, category,"masks/", self.dataset_csv.iloc[index, 0])
        # normalize image from 0-1 off the bat
        
        ## get image and mask and category integer 
        return_image = torch.tensor(io.imread(image_name))
        return_lung_mask = torch.tensor(io.imread(mask_name))
        return_category = self.string_to_label[category]
        ##map category to labels using dictionary


        return return_image, return_lung_mask, return_category
