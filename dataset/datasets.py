import os
import torch
import pandas as pd
from skimage import io,color
from torch.utils.data import Dataset
from torchvision import transforms
#this is dataset for lungs
class LungDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.root_dir = root_dir
        self.dataset_csv = pd.read_csv(csv_file)
        self.transform = transform  
        self.string_to_label = {
            'COVID': 0,
            'Normal': 1,
            'Viral Pneumonia': 0 # map these both to abnormal 
        }

    def get_sklearn_representation(self):
      data = []
      labels = []  
      for datum, mask, label in self:
        data.append(data)
        labels.append(label)
      return data, labels


    def __len__(self):
        return len(self.dataset_csv)

    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        category = self.dataset_csv.iloc[index, 1]
        # pull index from csv file
        image_name = os.path.join(self.root_dir, category,"images/", self.dataset_csv.iloc[index, 0])
        mask_name = os.path.join(self.root_dir, category,"masks/", self.dataset_csv.iloc[index, 0])
        
        ## get image and mask and category integer 
        return_image = io.imread(image_name)
        if len(return_image.shape) == 3 and return_image.shape[2] == 3:
            return_image = color.rgb2gray(return_image)
        return_image = torch.tensor(return_image) 
        return_image = self.transform(return_image)
    
        return_lung_mask = torch.tensor(color.rgb2gray(io.imread(mask_name)))
        return_image =torch.where( return_lung_mask == 1, return_image,0)
        return_image = (torch.tensor(color.gray2rgb(return_image)).permute(2,0,1)/255.0).float();
        return_category = self.string_to_label[category]        ##map category to labels using dictionary



        return return_image,return_lung_mask,  return_category
