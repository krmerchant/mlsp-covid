from dataset.datasets import LungDataset
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

data_dir = '/home/komelmerchant/Desktop/JHUCourseTracking/MachineLearningForSignalProcessing/project/data/COVID-19_Radiography_Dataset'
csv_file="./dataset/csv/small_covid_dataset.csv"
tf = transforms.Compose([transforms.CenterCrop(256), ]);
dataset = LungDataset(csv_file,data_dir,tf)

#train_loader = DataLoader(dataset, batch_size=16);
image,lung_mask,label = dataset[5]
image  = image.permute(1,2,0)

print(image)
print(image.shape)
print(label)
#display stuff
figure, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[1].imshow(lung_mask)


plt.show()

