# plot ROC AUC line up by distance and angle

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.lines as mlines

# Load the CSV file
df = pd.read_csv("master.csv")

# Map unique markers to unique 'angl' values
unique_angl = df['Angle_rad'].unique()
marker_map = {val: marker for val, marker in zip(unique_angl, ['o', 's', '^', 'D', 'v', 'X'])}

# Map unique colors to each feature
unique_features = df.loc[:, 'ASM':'variance'].columns
color_map = {feature: color for feature, color in zip(unique_features, plt.cm.tab10.colors)}

# Normalize the unique 'Angle_rad' values to the range [0, 1]
norm = plt.Normalize(vmin=0, vmax=len(unique_angl) - 1)

# Generate unique colors for each 'Angle_rad' using the viridis colormap
angle_color_map = {val: cm.viridis(norm(i)) for i, val in enumerate(unique_angl)}


# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))


# Iterate over each feature (columns I to R)
'''
for feature in df.loc[:, 'ASM':'variance'].columns:
    # Plot the line for the feature
    ax.plot(
        df['Dist_pix'],                # X-axis: Column G
        df['ROC_precise'],            # Y-axis: Feature weights
        label=feature,          # Label for the legend
        color=color_map[feature],         # Unique color for each feature
        marker=None,            # No marker for the line itself
    )
'''
'''
# Add markers for each point based on 'angl'
for i, row in df.iterrows():
    ax.plot(
        row['Dist_pix'],           # X-axis: Column G
        row['ROC_precise'],       # Y-axis: Feature weight
        color=angle_color_map[row['Angle_rad']],  # Unique color for each Angle_rad
        marker=marker_map[row['Angle_rad']],  # Marker shape based on 'angl'
        linestyle='-', 
        linewidth=1,
    )
'''

# Group the data by 'Angle_rad' and plot lines for each group
for angle, group in df.groupby('Angle_rad'):
    # Sort the group by the x-axis column 'Dist_pix'
    group = group.sort_values(by='Dist_pix')
    
    ax.plot(
        group['Dist_pix'],           # X-axis: Column Dist_pix
        group['ROC_precise'],        # Y-axis: Column ROC_precise
        color=angle_color_map[angle],  # Unique color for each Angle_rad
        marker=marker_map[angle],     # Marker shape based on Angle_rad
        linestyle='-',               # Line style
        linewidth=1,                 # Line width
        label=f"Angle_rad: {angle}"  # Label for the legend
    )


# Add labels and legend
ax.set_xlabel("Distance (pixels)")
ax.set_ylabel("ROC AUC")
ax.set_title("ROC AUC Lineup for SVM Model\nfrom Each Distance and Angle Combination")
#feature_legend = ax.legend(title="Features", fontsize='small', title_fontsize='medium', loc='upper left', bbox_to_anchor=(1, 1))

# Create a custom legend for Angle_rad markers
marker_legend_elements = [
    mlines.Line2D(
        [], [], 
        color=angle_color_map[angle], 
        marker=marker_map[angle], 
        linestyle='-', 
        markersize=8, 
        label=f"Angle_rad: {angle}"
    ) for angle in unique_angl
]

# Add the custom legend for Angle_rad
ax.legend(
    handles=marker_legend_elements,
    title="Angle_rad",
    fontsize='small',
    title_fontsize='medium',
    loc='upper left',
    bbox_to_anchor=(1, 0.3)  # Position below the feature legend
)

# Add the feature legend back to avoid overwriting
#ax.add_artist(feature_legend)

# Add faint horizontal gridlines
ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
# Format the x-axis with a step of 1
ax.set_xticks(np.arange(df['Dist_pix'].min(), df['Dist_pix'].max() + 1, 1))

# Show the plot
plt.tight_layout()
plt.show()


