from orchestrators.classifier_module import LitClassifier
from dataset.datasets import LungDataset
import click
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import lightning as L
from models.model import CustomConvNet
from pytorch_lightning.loggers import TensorBoardLogger
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Function to extract embeddings from the model
def extract_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    images = []
    labels = []
    with torch.no_grad():
        for data in data_loader:
            img,_, label = data
            img = img.to(device)
            embedding = torch.unsqueeze(torch.squeeze(model(img)),0)
            print(embedding.shape) 
            embeddings.append(embedding.cpu().numpy())
            images.append(torch.unsqueeze(img.flatten(),0).cpu().numpy())
            labels.append(label.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels,images

def plot_tsne_embeddings(embeddings:List, images:List, labels:List ):
    # Extract embeddings from the autoencoder model
    
    # Apply PCA to reduce the embeddings to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
   
   

    fig1, ax1 = plt.subplots(figsize=(8, 6))    # Plot the embeddings in 2D
    scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='Accent', alpha=0.6)
    ax1.set_title('2D TSNE Projection of Features Extracted from CNN')
    ax1.set_xlabel('TSNE Component 1')
    ax1.set_ylabel('TSNE Component 2')



    
    images_2d = tsne.fit_transform(images)
    fig2, ax2 = plt.subplots(figsize=(8, 6))    # Plot the images in 2D
    scatter = ax2.scatter(images_2d[:, 0], images_2d[:, 1], c=labels, cmap='Accent', alpha=0.6)
    ax2.set_title('2D TSNE Projection of Original X-Rays')
    ax2.set_xlabel('TSNE Component 1')
    ax2.set_ylabel('TSNE Component 2')


    plt.show()



@click.command()
@click.argument('checkpoint')
@click.argument('dataset')
@click.argument('datadir')
def generate_features(checkpoint, dataset,datadir):

    tf = transforms.Compose([transforms.CenterCrop(256), ]);
   
    logger.info("Loading dataset..") 
    dataset = LungDataset(dataset,datadir,tf)
    test_loader = DataLoader(dataset,batch_size=1);
   
    logger.info("Creating Model ...") 
    conv_net = CustomConvNet(num_classes=1)
    check = torch.load(checkpoint)
   

    model = LitClassifier.load_from_checkpoint(checkpoint,classifier=conv_net)

    #we chop of the final layer so we can extract features from it
    feature_extractor = nn.Sequential(*list(model.classifier.resnet.children())[:-1])
     
    embeddings, labels, images = extract_embeddings(feature_extractor, test_loader) 
    print(images.shape) 
    print(embeddings.shape) 
    plot_tsne_embeddings(embeddings,images,labels)

@click.command()
@click.argument('checkpoint')
@click.argument('dataset')
@click.argument('datadir')
def test(checkpoint, dataset,datadir):

    tf = transforms.Compose([transforms.CenterCrop(256), ]);
   
    logger.info("Loading dataset..") 
    dataset = LungDataset(dataset,datadir,tf)
    test_loader = DataLoader(dataset,batch_size=16);
   
    logger.info("Creatin Model ...") 
    conv_net = CustomConvNet(num_classes=1)

    logger.info("Initializing Trainer ...") 
    classifier  = LitClassifier(conv_net)
    trainer = L.Trainer()
    trainer.test(model=classifier, ckpt_path=checkpoint,dataloaders=test_loader)
    fig, ax  = classifier.train_roc.plot(score=True)     

    conv_net.resnet = nn.Sequential(*list(model.children())[:-1])
    extract_embeddings(conv_net.resnet, test_loader) 
    

    plt.show()



@click.command()
@click.option('--batch_size', default=16, help='batch size')
@click.argument('dataset')
@click.argument('datadir')
def train(batch_size, dataset,datadir):

    tf = transforms.Compose([transforms.CenterCrop(256), ]);
   
    logger.info("Loading dataset..") 
    dataset = LungDataset(dataset,datadir,tf)
    train_loader = DataLoader(dataset, batch_size=batch_size);
   
    logger.info("Creatin Model ...") 
    conv_net = CustomConvNet(num_classes=1)

    logger.info("Initializing Trainer ...") 
    classifier  = LitClassifier(conv_net)
    
    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger("logs/", name="my_model")

    # Set up trainer with the logger
    trainer = L.Trainer(logger=tb_logger, max_epochs=1000)
    trainer.fit(model=classifier,train_dataloaders=train_loader)



