import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv("vortex_labels.csv")

# Cap the max_vorticity to 1
df_capped_1 = df.copy()
df_capped_1['max_vorticity'] = df_capped_1['max_vorticity'].clip(upper=1)
df_capped_1['min_distance'] = df_capped_1['min_distance'].clip(lower=30)

# Filter out the "not relevant" label
df_filtered = df_capped_1[df_capped_1['label'] != 'not relevant']
filtered_labels = df_filtered['label'].unique()
filtered_label_colors = {label: color for label, color in zip(filtered_labels, plt.cm.get_cmap('tab10').colors)}

# Create the filtered plot
plt.figure(figsize=(10, 6))
for label in filtered_labels:
    for fish_status, marker in zip([True, False], ['o', 'x']):
        subset = df_filtered[(df_filtered['label'] == label) & (df_filtered['fish'] == fish_status)]
        plt.scatter(
            1/(subset['min_distance']*subset['min_distance']),
            subset['max_vorticity'],
            label=f"{label}, fish={fish_status}",
            marker=marker,
            color=filtered_label_colors[label],
            alpha=0.7
        )

# Add labels and legend
plt.xlabel("1 / min_distance²")
plt.ylabel("max_vorticity (capped at 1)")
plt.title("Vortex Labels (excluding 'not relevant'): min_distance vs. max_vorticity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Count occurrences of each combination of label and fish status after filtering
label_fish_counts = df_filtered.groupby(['label', 'fish']).size().reset_index(name='count')
print(label_fish_counts)
