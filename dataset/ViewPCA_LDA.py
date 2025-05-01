import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import time
from ConstructFeatureMatrix import construct_feature_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
import mpl_toolkits.mplot3d  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

superkeys = {
    "contrast",
    "energy",
    "entropy",
    "correlation",
    "dissimilarity",
    "homogeneity",
    "ASM",
    "mean",
    "variance",
    "std",
    "datasetID",
    "imageName",
    "label1",
    "label2",
    "dist",
    "angl"
}
removedkeys = {
    "datasetID",
    "imageName",
    "label1",
    "label2",
    "dist",
    "angl"
}

#orderCombo = ['dist_1_angl_1', 'dist_1_angl_2', 'dist_2_angl_1']
#orderCombo = ['test_dist_1_angl_1', 'test_dist_1_angl_2']
orderCombo = ['test_dist_3_angl_2']

folderName = '-'.join(orderCombo)
folderName = folderName.replace('test', 't').replace('dist', 'd').replace('angl', 'a')
# Define the path where you want to create the folder
folder_path = Path(folderName + '_LDA_scaled')  # for specific distance and angle
# Create the folder
folder_path.mkdir(parents=True, exist_ok=True)

feature_data, label1_data, label2_data, newkeys, filenames, num_features, num_json_files = construct_feature_matrix(superkeys, removedkeys, orderCombo)
features = newkeys


# Combine feature_data and label1_data into a DataFrame
df = pd.DataFrame(feature_data, columns=features)  # Use feature names as column names
df['label'] = label1_data.flatten()  # Add label1_data as a new column
# Use sns.pairplot to visualize the relationships
sns.pairplot(df, hue='label', diag_kind='kde', palette='Set2')
#plt.show()
plt.savefig(Path(folder_path, 'PairPlot.png'))

'''
# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
pca_result = pca.fit_transform(feature_data)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['label'] = label1_data.flatten()  # Add labels for coloring

# Plot PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', palette='Set2', s=50)
plt.title('PCA Visualization (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Label')
plt.grid(True)
plt.show()

# Perform PCA for 3D visualization
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(feature_data)

# Create a DataFrame for PCA results
pca_3d_df = pd.DataFrame(pca_result_3d, columns=['PC1', 'PC2', 'PC3'])
pca_3d_df['label'] = label1_data.flatten()  # Add labels for coloring

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    pca_3d_df['PC1'], pca_3d_df['PC2'], pca_3d_df['PC3'],
    c=pca_3d_df['label'], cmap='Set2', s=50
)
ax.set_title('PCA Visualization (3D)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
fig.colorbar(scatter, ax=ax, label='Label')
plt.show()
'''
# Scale the features
scaler = StandardScaler()
scaled_feature_data = scaler.fit_transform(feature_data)

# Perform LDA
lda = LDA(n_components=1)  # Reduce to 2 dimensions for visualization
lda_result = lda.fit_transform(scaled_feature_data, label1_data.flatten())

# Create a DataFrame for LDA results
lda_df = pd.DataFrame(lda_result, columns=['LD1'])
lda_df['label'] = label1_data.flatten()  # Add labels for coloring

# Plot LDA results as a strip plot
plt.figure(figsize=(8, 6))
sns.stripplot(data=lda_df, x='label', y='LD1', palette='Set2', jitter=True, size=8)
plt.title('LDA Visualization (1D)')
plt.xlabel('Label')
plt.ylabel('Linear Discriminant 1')
plt.grid(True)
#plt.show()
plt.savefig(Path(folder_path, 'LDAstrip.png'))


# Plot LDA results as a histogram
plt.figure(figsize=(8, 6))
for label in np.unique(label1_data):
    subset = lda_df[lda_df['label'] == label]
    plt.hist(subset['LD1'], bins=20, alpha=0.5, label=f'Class {label}')

plt.title('LDA Visualization (1D)')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Frequency')
plt.legend(title='Label')
plt.grid(True)
#plt.show()
plt.savefig(Path(folder_path, 'LDAhist.png'))


# Perform LDA
lda = LDA()
lda.fit(feature_data, label1_data.flatten())

# Extract feature importance (coefficients)
feature_importance = np.abs(lda.coef_[0])  # Absolute values of coefficients for the first linear discriminant

# Rank features by importance
feature_ranking = np.argsort(feature_importance)[::-1]  # Indices of features sorted by importance (descending)
ranked_features = [newkeys[i] for i in feature_ranking]  # Map indices to feature names

# Print ranked features and their importance
print("Feature Ranking (by importance):")
for i, feature in enumerate(ranked_features):
    print(f"{i+1}. {feature}: {feature_importance[feature_ranking[i]]}")

# Select top k features (e.g., top 3)
k = 3
top_features = ranked_features[:3]
print(f"\nTop {k} features: {top_features}")

# Define the file path for saving feature ranking
feature_ranking_file_path = Path(folder_path, "feature_ranking.txt")
# Write feature ranking to the file
with open(feature_ranking_file_path, "w") as file:
    file.write("Feature Ranking (by importance):\n")
    for i, feature in enumerate(ranked_features):
        file.write(f"{i+1}. {feature}: {feature_importance[feature_ranking[i]]}\n")
