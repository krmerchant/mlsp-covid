import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
#this is dataset for lungs
class LungDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.root_dir = root_dir
        self.dataset_csv = pd.read_csv(csv_file)
        self.transform = transforms.CenterCrop(256)
        self.string_to_label = {
            'COVID': 1,
            'Normal': 0,
            'Viral Pneumonia': 1
            #'Lung_Opacity": 1, would need to include in csv file
        }
        self.string_to_label2 = {
            'COVID': 1,
            'Normal': 0,
            'Viral Pneumonia': 2
            #'Lung_Opacity": 3, would need to include in csv file
        }
    '''
    def get_category_map(self):
       category_map = {1: 'COVID', 0: 'Normal', 1: 'Viral Pneumonia', 1: 'Lung_Opacity'}
       return category_map
    '''

    def get_sklearn_representation(self):
      data = []
      labels = []  
      for datum, mask, label in self:
        data.append(datum)
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
        return_image = torch.tensor(io.imread(image_name))
        if len(return_image.shape) == 3: # if image is RGB, convert to grayscale, assume r=g=b
            return_image = return_image[:, :, 0]
        return_image = self.transform(return_image)
        return_lung_mask = torch.tensor(io.imread(mask_name))
        return_category = self.string_to_label[category]
        return_category2 = self.string_to_label2[category]
        ##map category to labels using dictionary


        # apply mask to image
        #convert lung mask from RGB size [255,255,3] to grayscale size [255,255] to match image's grayscale size
        #already checked that each pixel is either [0,0,0] or [255,255,255] in lung_mask. 
        #thus, will disregard the 2nd and 3rd color channels and only take the first color channel
        #so go thru each pixel, and slice/take only the 1st color channel value (index 0)
        slice = torch.tensor([[return_lung_mask[i][j][0] for j in range(256)] for i in range(256)])
        return_masked_image = return_image & slice  

        filename = self.dataset_csv.iloc[index, 0]
        return return_image, return_lung_mask, return_category, return_category2, return_masked_image, filename
    
       
