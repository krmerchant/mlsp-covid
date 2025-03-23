from orchestrators.classifier_module import LitClassifier
from dataset.datasets import LungDataset
import click
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import lightning as L
from models.model import CustomConvNet
from pytorch_lightning.loggers import TensorBoardLogger


#logging setup crap
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter(
    '[%(levelname)s:  %(asctime)s] - %(name)s  - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(ch)

@click.command()
@click.option('--batch_size', default=2, help='batch size')
@click.option('--dataset',  default="./dataset/csv/small_covid_dataset.csv")
@click.option('--datadir',default='/home/komelmerchant/Desktop/JHUCourseTracking/MachineLearningForSignalProcessing/project/data/COVID-19_Radiography_Dataset')
def train(batch_size, dataset,datadir):

    tf = transforms.Compose([transforms.CenterCrop(256), ]);
   
    logger.info("Loading dataset..") 
    dataset = LungDataset(dataset,datadir,tf)
    train_loader = DataLoader(dataset, batch_size=batch_size);
   
    logger.info("Creatin Model ...") 
    conv_net = CustomConvNet(num_classes=2)

    logger.info("Initializing Trainer ...") 
    classifier  = LitClassifier(conv_net)
    
    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger("logs/", name="my_model")

    # Set up trainer with the logger
    trainer = L.Trainer(logger=tb_logger, max_epochs=1000)
    trainer.fit(model=classifier,train_dataloaders=train_loader)
    
  # Check if the script is being run directly
if __name__ == "__main__":
    train()



