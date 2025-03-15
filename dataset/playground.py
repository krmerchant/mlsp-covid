from datasets import LungDataset
from matplotlib import pyplot as plt
import torch
 
 
 
def main():
    # Create an instance of the LungDataset class
   # dataset = LungDataset(csv_file='C:\Users\Dory\Documents\GitHub\mlsp-covid\dataset\csv\covid_dataset.csv',
    dataset = LungDataset(csv_file=r'C:\Users\Dory\Documents\GitHub\mlsp-covid\dataset\csv\covid_dataset.csv',
                          root_dir=r'C:\Users\Dory\Documents\JHU\Machine Learning for Signal Processing\COVID-19_Radiography_Dataset')    
    print(len(dataset))
    image, lung_mask, label, masked_image = dataset[12]

    #convert mask from RGB size to grayscale size
    #slice = torch.tensor([[lung_mask[i][j][0] for j in range(256)] for i in range(256)])
    #type(slice)
    #slice.shape
    #apply mask
    #masked_image = torch.tensor([[slice[i][j] & image[i][j] for j in range(256)] for i in range(256)])
    #masked_image = torch.tensor([[slice[i][j] and image[i][j] for j in range(256)] for i in range(256)])
    #masked_image = image * slice
    #masked_image = image & slice
    #display stuff
    print(label)
    print(dataset.get_category_map())
    #mkplt(plt,image,lung_mask,slice,masked_image)
    mkplt2(plt,image,lung_mask,masked_image)

    #dataset.get_sklearn_representation()

def mkplt(plt,image,lung_mask,slice,masked_image):
    figure, axes = plt.subplots(1, 4)
    axes[0].imshow(image)
    axes[1].imshow(lung_mask)
    axes[2].imshow(slice)
    axes[3].imshow(masked_image)
    plt.show()

def mkplt2(plt,image,lung_mask,masked_image):
    figure, axes = plt.subplots(1, 3)
    axes[0].imshow(image)
    axes[1].imshow(lung_mask)
    axes[2].imshow(masked_image)
    plt.show()

if __name__ == "__main__":
    main()