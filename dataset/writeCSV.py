import csv

# Data to write
data = [
    ["Name", "Age", "City"],
    ["Alice", 30, "New York"],
    ["Bob", 25, "Los Angeles"],
    ["Charlie", 35, "Chicago"]
]

# Filepath to save the CSV
output_file = "output.csv"

# Write to CSV
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Data written to {output_file}")

# Read the CSV file to verify

import pandas as pd

df = pd.read_csv(output_file)
print("Contents of the CSV file:")
print(df)