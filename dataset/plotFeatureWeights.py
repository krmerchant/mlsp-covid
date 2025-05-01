import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers

import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting module

# Load the CSV file
df = pd.read_csv("combined_features_weights_all_ang_d1_d16_C1.csv")

# Map marker shapes (customize as needed)
unique_markers = df['angl'].unique()
marker_map = {val: marker for val, marker in zip(unique_markers, ['o', 's', '^', 'D', 'v', 'X'])}

# Map colors to unique features
unique_features = df['Feature'].unique()
color_map = {feature: color for feature, color in zip(unique_features, plt.cm.tab10.colors)}

# Create the plot
fig, ax = plt.subplots()

# Group by 'Feature' (color) and 'angl' (marker shape)
for (feature_label, marker_label), group in df.groupby(['Feature', 'angl']):
    ax.scatter(
        group['Weight'],     # x-axis
        group['dist_pix'],   # y-axis
        label=f"{feature_label} | {marker_label}",  # Combine labels for legend
        color=color_map[feature_label],            # Color based on 'Feature'
        marker=marker_map[marker_label],           # Marker shape based on 'angl'
    )

# Add labels to the main plot
ax.set_xlabel("Weight")
ax.set_ylabel("dist_pix")

# Do not display the legend in the first figure
# Create a separate figure for the legend
fig_legend, ax_legend = plt.subplots(figsize=(6, 4))  # Adjust size as needed
handles, labels = ax.get_legend_handles_labels()  # Extract legend handles and labels
ax_legend.axis('off')  # Turn off the axis for the legend-only figure
ax_legend.legend(handles, labels, loc='center', title="Feature (Color) | Angle (Marker)",
                 fontsize='small',          # Reduce font size of legend labels
                 title_fontsize='medium',   # Adjust font size of the legend title
                 labelspacing=0.5,          # Reduce spacing between labels
                 handlelength=1.5)          # Adjust the length of the legend handles

# Show both figures
plt.tight_layout()
plt.show()




'''
# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

# Group by 'Feature' (color) and 'angl' (marker shape)
for (feature_label, marker_label), group in df.groupby(['Feature', 'angl']):
    ax.scatter(
        group['Weight'],     # x-axis
        group['dist_pix'],   # y-axis
        group['angl'],       # z-axis
        label=f"{feature_label} | {marker_label}",  # Combine labels for legend
        color=color_map[feature_label],            # Color based on 'Feature'
        marker=marker_map[marker_label],           # Marker shape based on 'angl'
    )

# Add labels to the axes
ax.set_xlabel("Weight")
ax.set_ylabel("dist_pix")
ax.set_zlabel("angl")


# Show the plot
plt.tight_layout()
plt.show()
'''
# Get unique values of 'angl'
unique_angl = df['angl'].unique()
marker_map = {val: marker for val, marker in zip(unique_angl, ['o', 's', '^', 'D', 'v', 'X'])}

# Create stacked subplots
fig, axes = plt.subplots(nrows=len(unique_angl), ncols=1, figsize=(8, len(unique_angl) * 4), sharex=True)

# Iterate over each unique 'angl' and corresponding subplot
for ax, angl_value in zip(axes, unique_angl):
    # Filter the DataFrame for the current 'angl'
    df_subset = df[df['angl'] == angl_value]
    
    # Group by 'Feature' and plot
    for feature_label, group in df_subset.groupby('Feature'):
        ax.scatter(
            group['Weight'],     # x-axis
            group['dist_pix'],   # y-axis
            label=feature_label,  # Label for the legend
            color=color_map[feature_label],  # Color based on 'Feature'
            marker=marker_map[angl_value] # Marker shape based on 'angl'
        )
    
    # Add labels and title for the subplot
    ax.set_title(f"angl = {angl_value}")
    ax.set_ylabel("dist_pix")
    #ax.legend(title="Feature", fontsize='small', title_fontsize='medium')

# Add a shared x-axis label
fig.text(0.5, 0.04, "Weight", ha='center', fontsize=12)

# Adjust layout
plt.tight_layout()
plt.show()
