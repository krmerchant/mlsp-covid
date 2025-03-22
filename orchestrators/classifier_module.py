import torch
import torch.nn as nn
import lightning as L


class LitClassifier(L.LightningModule):
    def __init__(self, classifier:nn.Module):
        super().__init__()
        self.classifier = classifier 
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data,_,labels = batch
        output = self.classifier(data)
        loss = nn.CrossEntropyLoss(output,labels) 
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, momentum=0.9)
