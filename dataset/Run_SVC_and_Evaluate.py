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
import csv
from sklearn.preprocessing import StandardScaler
import pandas as pd


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

'''
removedkeys = {
    "contrast",
    "entropy",
    "dissimilarity",
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
'''
#distIdx = [3]
#anglIdx = [1,2,3,4]
#distIdx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#anglIdx = [1,2,3,4]

#for iii in range(len(distIdx)):
    #for jjj in range(len(anglIdx)):

distingish_folder = False #to try different feature reduction
distinguish_folder_str = 'v2'

#orderCombo = ['dist_1_angl_1', 'dist_1_angl_2', 'dist_2_angl_1']
#orderCombo = ['test_dist_3_angl_1', 'test_dist_3_angl_2', 'test_dist_3_angl_3','test_dist_3_angl_4','test_dist_5_angl_1', 'test_dist_5_angl_2', 'test_dist_5_angl_3','test_dist_5_angl_4','test_dist_2_angl_1', 'test_dist_2_angl_2', 'test_dist_2_angl_3','test_dist_2_angl_4','test_dist_8_angl_1', 'test_dist_8_angl_2', 'test_dist_8_angl_3','test_dist_8_angl_4','test_dist_4_angl_1', 'test_dist_4_angl_2', 'test_dist_4_angl_3','test_dist_4_angl_4','test_dist_1_angl_1', 'test_dist_1_angl_2', 'test_dist_1_angl_3','test_dist_1_angl_4','test_dist_6_angl_1', 'test_dist_6_angl_2', 'test_dist_6_angl_3','test_dist_6_angl_4','test_dist_7_angl_1', 'test_dist_7_angl_2', 'test_dist_7_angl_3','test_dist_7_angl_4','test_dist_9_angl_1', 'test_dist_9_angl_2', 'test_dist_9_angl_3','test_dist_9_angl_4','test_dist_10_angl_1', 'test_dist_10_angl_2', 'test_dist_10_angl_3','test_dist_10_angl_4','test_dist_11_angl_1', 'test_dist_11_angl_2', 'test_dist_11_angl_3','test_dist_11_angl_4','test_dist_12_angl_1', 'test_dist_12_angl_2', 'test_dist_12_angl_3','test_dist_12_angl_4','test_dist_13_angl_1', 'test_dist_13_angl_2', 'test_dist_13_angl_3','test_dist_13_angl_4','test_dist_14_angl_1', 'test_dist_14_angl_2', 'test_dist_14_angl_3','test_dist_14_angl_4','test_dist_15_angl_1', 'test_dist_15_angl_2', 'test_dist_15_angl_3','test_dist_15_angl_4','test_dist_16_angl_1', 'test_dist_16_angl_2', 'test_dist_16_angl_3','test_dist_16_angl_4']
orderCombo = ['full_dist_3_angl_2']

#orderCombo = [f'test_dist_{distIdx[iii]}_angl_{anglIdx[jjj]}']

C_param = 1  # Regularization parameter
kernel_param = 'linear'  # Kernel type to be used in the algorithm
#gamma_param = 'scale'  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
# Create a string for the file name
svc_params_string = f"_SVM_C_{C_param}_k_{kernel_param}".replace('.', '_')

folderName = '-'.join(orderCombo)
#add 'n_' to the folder name to distinguish between normal and normalized features!!!!!!!!!!!!!
folderName = folderName.replace('full', 'n_f').replace('dist', 'd').replace('angl', 'a')
#folderName = 'all_ang_d1_to_d16' #for specific distance and angle
# Define the path where you want to create the folder
folder_path = Path(folderName + svc_params_string)  # for specific distance and angle
if distingish_folder:
    folder_path = Path(folderName + svc_params_string + '_' + distinguish_folder_str) 
# Create the folder
folder_path.mkdir(parents=True, exist_ok=True)

feature_data, label1_data, label2_data, newkeys, filenames, num_features, num_json_files = construct_feature_matrix(superkeys, removedkeys, orderCombo)
features = newkeys

test_size_percentage = 0.3  # Percentage of the dataset to be used as test set
X_train, X_test, y_train, y_test = train_test_split(feature_data, label1_data, test_size=test_size_percentage, random_state=42)
# for debugging (overfitting)
# X_test = X_train
# y_test = y_train

#Scaling the features
# Initialize the scaler
scaler = StandardScaler()
# Fit and transform the training data
X_train = scaler.fit_transform(X_train)
# Transform the test data using the same scaler
X_test = scaler.transform(X_test)



# Create an instance of the Support Vector Classifier (SVC) with a linear kernel
model = SVC(C = C_param, kernel = kernel_param , class_weight='balanced', probability=True) # Use class_weight='balanced' to handle class imbalance


# Define the JSON file path inside the folder
json_file_path = Path(folder_path, "features.json")
parameters = {"features": features, "C": C_param, "kernel": kernel_param, "orderCombo": orderCombo}

# Write dictionary to a file
with open(json_file_path, "w") as file:
    json.dump(parameters, file, indent=4)  # Use indent=4 for pretty formatting


start_time = time.time()  # Start the timer

# Train the model on the training data
model.fit(X_train, y_train)

end_time = time.time()  # End the timer
print(f"Script runtime: {end_time - start_time:.2f} seconds")



# Get weights (coefficients) and intercept
weights = model.coef_
intercept = model.intercept_
numSupportVectors = len(model.support_)

# Define the file path for saving weights and intercept
weights_file_path = Path(folder_path, "weights_and_intercept.txt")

# Save weights, intercept, and associated features to a text file
with open(weights_file_path, "w") as file:
    # Save num support vectors
    file.write("Number of Support Vectors:\n")
    file.write(f"{numSupportVectors}\n\n")  # Convert to list for better formatting

    # Save intercept
    file.write("Intercept (bias):\n")
    file.write(f"{intercept.tolist()}\n\n")  # Convert to list for better formatting

    # Save weights and associated features
    file.write("Feature Weights:\n")
    for ccc, class_weights in enumerate(weights):  # For multi-class classification
        file.write(f"Class {ccc}:\n")
        for feature, weight in zip(features, class_weights):
            file.write(f"{feature}: {weight}\n")  # Save weights rounded to 4 decimal places
        file.write("\n")


# Combine features and orderCombo
combined_features = [f"{feature}_{order}" for feature in features for order in orderCombo]

# Define the CSV file path for saving feature weights
weights_csv_path = Path(folder_path, "combined_features_weights.csv")

# Save combined features and weights to a CSV file
with open(weights_csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(["Combined Feature", "Weight", "Class"])  # Add "Class" for multi-class classification

    # Write the weights for each class
    for ccc, class_weights in enumerate(weights):  # For multi-class classification
        for combined_feature, weight in zip(combined_features, class_weights):

            split_parts = combined_feature.split("_")  # Split the string by '_'

            writer.writerow([combined_feature, weight, f"Class {ccc}"]+split_parts)  # Save weights rounded to 4 decimal places


# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print accuracy score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))  # Print detailed classification report


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)




# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
#plt.show()
plt.savefig(Path(folder_path, 'ROC.png'))


#conf_matrix = confusion_matrix(labels, predictions, labels=clf.classes_)

cmn = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=['Abnormal Lungs', 'Healthy Lungs'], yticklabels=['Abnormal Lungs', 'Healthy Lungs'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
#plt.show()
plt.title("Confusion Matrix")
plt.savefig(Path(folder_path, 'CM.png'))


#round values
# Round confusion matrix percentages to 2 decimal places
cmn_round = np.round(cmn, 2)
# Round ROC AUC value to 2 decimal places
roc_auc_round = round(roc_auc, 2)


# Save classification report and confusion matrix to a file
report_file_path = Path(folder_path, "report_cm.txt")

with open(report_file_path, "w") as file:
    # Write test_size and dataset sizes
    file.write(f"Test Size: {test_size_percentage * 100:.0f}%\n")
    file.write(f"Number of files in X_train: {len(X_train)}\n")
    file.write(f"Number of files in X_test: {len(X_test)}\n\n")

    # Save classification report
    classification_rep = classification_report(y_test, y_pred)
    file.write("Classification Report:\n")
    file.write(classification_rep)
    file.write("\n\n")

    # Save confusion matrix
    file.write("Confusion Matrix:\n")
    file.write(np.array2string(conf_matrix, separator=', '))
    file.write("\n\n")

    # Save confusion matrix percentage
    file.write("Confusion Matrix Percentage:\n")
    file.write(np.array2string(cmn, separator=', '))
    file.write("\n\n")

    # Save confusion matrix percentage round
    file.write("Confusion Matrix Percentage:\n")
    file.write(np.array2string(cmn_round, separator=', '))
    file.write("\n\n\n")

    # Save ROC AUC value
    file.write("ROC AUC:\n")
    file.write(np.array2string(roc_auc))
    file.write("\n\n")

    # Save ROC AUC value
    file.write("ROC AUC:\n")
    file.write(np.array2string(roc_auc_round))
    file.write("\n")

#write to csv file
track_folder_roc = [str(folder_path), roc_auc, roc_auc_round]
track_filename_roc = "test_FolderPath_ROC.csv"
# Append the new row to the CSV file
with open(track_filename_roc, mode="a", newline="") as file:  # Open in append mode
    writer = csv.writer(file)
    writer.writerow(track_folder_roc)


#write to csv file - feature weights
track_folder_features = [str(folder_path), intercept, weights]
track_filename_features = "test_FolderPath_FeatureWeights.csv"
# Append the new row to the CSV file
with open(track_filename_features, mode="a", newline="") as file:  # Open in append mode
    writer = csv.writer(file)
        # Write the weights for each class
    for ccc, class_weights in enumerate(weights):  # For multi-class classification
        row = [str(folder_path), intercept[ccc]] + class_weights.tolist() + [numSupportVectors] # Convert weights to a list
        writer.writerow(row)