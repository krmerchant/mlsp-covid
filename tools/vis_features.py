import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
# Set device`
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to extract embeddings from the model
def extract_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in data_loader:
            img, label = data
            img = img.to(device)
            _, embedding = model(img)
            embeddings.append(embedding.cpu().numpy())
            labels.append(label.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels


def plot_pca_embeddeings(embeddings, labels):
  # Apply PCA to reduce the embeddings to 2D
  pca = PCA(n_components=2)
  embeddings_2d = pca.fit_transform(embeddings)

  # Plot the embeddings in 2D
  plt.figure(figsize=(10, 8))
  scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
  plt.colorbar(scatter)
  plt.title('2D PCA Projection of Embeddings')
  plt.xlabel('PCA Component 1')
  plt.ylabel('PCA Component 2')
  plt.show()


