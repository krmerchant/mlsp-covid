from datasets import LungDataset
from matplotlib import pyplot as plt
import torch
from skimage import io
from skimage.feature import graycomatrix, graycoprops
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
if torch.cuda.is_available():
    print("Using GPU")
dataset = LungDataset(csv_file=r'C:\Users\Dory\Documents\GitHub\mlsp-covid\dataset\csv\covid_dataset.csv', root_dir=r'C:\Users\Dory\Documents\JHU\Machine Learning for Signal Processing\COVID-19_Radiography_Dataset')    
dist = np.array([5]) 
ang= np.array([0])
features, labels = extract_GLCM_features(dataset, dist, ang)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    if torch.cuda.is_available():
        print("Using GPU")

    # Create an instance of the LungDataset class
    dataset = LungDataset(csv_file=r'C:\Users\Dory\Documents\GitHub\mlsp-covid\dataset\csv\covid_dataset.csv', root_dir=r'C:\Users\Dory\Documents\JHU\Machine Learning for Signal Processing\COVID-19_Radiography_Dataset')    
    #print(len(dataset))

    image, lung_mask, label, masked_image = dataset[12]
    #print(label)

    #distances = [1, 5, 25]  # Distance between pixels
    dist = np.array([5]) 
    #angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135 deg
    #angles = [0, np.pi/2]  #horizontal and vertical
    ang= np.array([0])  






    features, labels = extract_GLCM_features(dataset, dist, ang)

'''
    GLCM = graycomatrix(masked_image, dist, ang)
    prop_contrast = graycoprops(GLCM, 'contrast')
    prop_energy = graycoprops(GLCM, 'energy')
    prop_entropy = graycoprops(GLCM, 'entropy')
    prop_correlation = graycoprops(GLCM, 'correlation')
    prop_dissimilarity = graycoprops(GLCM, 'dissimilarity')
    prop_homogeneity = graycoprops(GLCM, 'homogeneity')

    print(GLCM)
    print(GLCM.shape)
    print(type(GLCM))
'''
'''
    features, labels_to_features = extract_GLCM_features(dataset, dist, ang)
    print(features.shape)
    print(labels_to_features.shape)


    mkplt(plt,image,lung_mask,masked_image)

'''


def mkplt(plt,image,lung_mask,masked_image):
    figure, axes = plt.subplots(2, 2)
    axes[0,0].imshow(image,cmap='gray')
    axes[0,1].imshow(lung_mask)
    axes[1,0].imshow(masked_image,cmap='gray')
    axes[1,1].imshow(masked_image)
    plt.show()


def extract_GLCM_features(dataset, dist, ang):
    num_props = 6  # Number of GLCM properties to extract
    return_features = np.empty((len(dataset), len(dist) * len(ang) * num_props))  # Adjust shape based on features
    return_labels = np.empty(len(dataset))

    #for i in range(len(dataset)):
    for i in range(int(np.array(1))):
        image, lung_mask, label, masked_image = dataset[i]
        #masked_image_tensor = torch.tensor(masked_image, device=device)
        #dist_tensor = torch.tensor(dist_tensor, device=device)
        #ang_tensor = torch.tensor(ang_tensor, device=device)
        
        GLCM = graycomatrix(masked_image, dist, ang)

        prop_1 = graycoprops(GLCM, 'contrast')
        prop_2 = graycoprops(GLCM, 'energy')
        prop_3 = graycoprops(GLCM, 'entropy')
        prop_4 = graycoprops(GLCM, 'correlation')
        prop_5 = graycoprops(GLCM, 'dissimilarity')
        prop_6 = graycoprops(GLCM, 'homogeneity')
                
        return_features[i,:] = np.concatenate((prop_1.flatten(), prop_2.flatten(), prop_3.flatten(), 
                                             prop_4.flatten(), prop_5.flatten(), prop_6.flatten()))
        return_labels[i] = np.array(label)

    return return_features, return_labels



if __name__ == "__main__":
    main()