import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import BinaryAccuracy

class LitClassifier(L.LightningModule):
    def __init__(self, classifier:nn.Module):
        super().__init__()
        self.classifier = classifier 
        self.loss_fn = nn.CrossEntropyLoss() 
        self.train_acc = BinaryAccuracy() 
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data,_,labels = batch
        output = self.classifier(data)
        loss = self.loss_fn(output,labels)
        self.log('train_loss',loss, on_step=False, on_epoch=True) 
        acc = self.train_acc(output,labels) 
        self.log('train_acc',acc, on_step=False, on_epoch=True) 
        
        print(loss.item()) 
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
