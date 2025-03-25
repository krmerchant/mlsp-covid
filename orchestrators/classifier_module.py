import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import BinaryAccuracy, BinarySpecificity, BinaryROC

import matplotlib.pyplot as plt
import io
import PIL.Image
import numpy as np

class LitClassifier(L.LightningModule):
    def __init__(self, classifier:nn.Module):
        super().__init__()
        self.classifier = classifier 
        self.loss_fn = nn.BCELoss() 
        self.train_acc = BinaryAccuracy() 
        self.train_spec = BinarySpecificity() 
        self.train_roc = BinaryROC(thresholds=10) 
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data,_,labels = batch
        labels =torch.unsqueeze(labels.float(),1) 
        output = self.classifier(data)
        loss = self.loss_fn(output,labels)
        self.log('train_loss',loss, on_step=False, on_epoch=True) 
        acc = self.train_acc(output,labels) 
        self.log('train_acc',acc, on_step=False, on_epoch=True) 
        spec = self.train_spec(output,labels)     
        self.log('train_spec',spec, on_step=False, on_epoch=True) 
        self.train_roc.update(output,labels.long()) 
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

