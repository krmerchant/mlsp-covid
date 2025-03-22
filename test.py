from dataset.datasets import LungDataset
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

data_dir = '/home/komelmerchant/Desktop/JHUCourseTracking/MachineLearningForSignalProcessing/project/data/COVID-19_Radiography_Dataset'
csv_file="./dataset/csv/covid_dataset.csv"


tf = transforms.Compose([transforms.CenterCrop(256), ]);
dataset = LungDataset(csv_file,data_dir,tf)

#for i, (x,y) in enumerate(dataset):
#    print(i)
#    print(x.shape)
#    print(y)
#
train_loader = DataLoader(dataset, batch_size=2);


#for (images,_,labels) in train_loader:
#    print(labels.shape)
#    print(images.shape)

image, lung_mask, label  = dataset[14692]
print(image.shape)
print(label)
#display stuff
figure, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[1].imshow(lung_mask)



plt.show()

