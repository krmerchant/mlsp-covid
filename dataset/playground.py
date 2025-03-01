from datasets import LungDataset
from matplotlib import pyplot as plt
 
 
 
def main():
    # Create an instance of the LungDataset class
   # dataset = LungDataset(csv_file='C:\Users\Dory\Documents\GitHub\mlsp-covid\dataset\csv\covid_dataset.csv',
    dataset = LungDataset(csv_file=r'C:\Users\Dory\Documents\GitHub\mlsp-covid\dataset\csv\covid_dataset.csv',
                          root_dir=r'C:\Users\Dory\Documents\JHU\Machine Learning for Signal Processing\COVID-19_Radiography_Dataset')    
    image, lung_mask, label = dataset[1]
    #display stuff
    figure, axes = plt.subplots(1, 3)
    axes[0].imshow(image)
    axes[1].imshow(lung_mask)
    print(label)
    plt.show()
if __name__ == "__main__":
    main()