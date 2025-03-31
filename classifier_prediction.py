from orchestrators.classifier_module import LitClassifier
from dataset.datasets import LungDataset
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import click
from torchvision import transforms
import logging
import lightning as L
from models.model import CustomConvNet
import matplotlib.pyplot as plt
from typing import List
import pickle


from sklearn import svm
import torch.nn as nn
import torch
import numpy as np
#logging setup crap
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter(
    '[%(levelname)s:  %(asctime)s] - %(name)s  - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(ch)

def extract_embeddings(model, data_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    embeddings = []
    images = []
    labels = []
    with torch.no_grad():
        for data in data_loader:
            img,_, label = data
            img = img.to(device)
            embedding = torch.unsqueeze(torch.squeeze(model(img)),0)
            embeddings.append(embedding.cpu().numpy())
            images.append(torch.unsqueeze(img.flatten(),0).cpu().numpy())
            labels.append(label.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels,images




@click.command()
@click.argument('checkpoint')
@click.argument('dataset')
@click.argument('datadir')
def train_svm(checkpoint, dataset,datadir):

    tf = transforms.Compose([transforms.CenterCrop(256), ]);
   
    logger.info("Loading dataset..") 
    dataset = LungDataset(dataset,datadir,tf)
    test_loader = DataLoader(dataset,batch_size=64);
   
    logger.info("Creatin Model ...") 
    conv_net = CustomConvNet(num_classes=1)

    model = LitClassifier.load_from_checkpoint(checkpoint,classifier=conv_net)
    feature_extractor = nn.Sequential(*list(model.classifier.resnet.children())[:-1])
   
    embeddings, labels, images = extract_embeddings(feature_extractor, test_loader) 
    clf = svm.SVC()
    embeddings = np.squeeze(embeddings)
    clf.fit(np.squeeze(embeddings), labels)

    predictions = clf.predict(np.squeeze(embeddings))
    cm = confusion_matrix(labels, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Abnormal Lungs', 'Healthy Lungs'])

    disp.plot()
    plt.show()

