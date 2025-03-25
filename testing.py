import matplotlib.pyplot as plt
from orchestrators.classifier_module import LitClassifier
from dataset.datasets import LungDataset
import click
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import lightning as L
from models.model import CustomConvNet
import logging
#logging setup crap
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter(
    '[%(levelname)s:  %(asctime)s] - %(name)s  - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(ch)




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
    plt.show()

