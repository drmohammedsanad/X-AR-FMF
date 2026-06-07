import matplotlib.pyplot as plt

# Data
datasets = ['Amazon', 'Yelp', 'Douban']
coverage = [0.0282, 0.8760, 0.0967]

# Create figure
plt.figure(figsize=(7, 5))

# Bar chart
bars = plt.bar(
    datasets,
    coverage,
    color=['#4C72B0', '#55A868', '#C44E52'],
    edgecolor='black'
)

# Labels
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Explainability Coverage', fontsize=12)
plt.title('Explainability Coverage Across Datasets', fontsize=13)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.02,
        f'{height:.4f}',
        ha='center',
        fontsize=10
    )

# Grid
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Layout
plt.tight_layout()

# Save figure
plt.savefig('coverage_plot.png', dpi=300)
plt.savefig('coverage_plot.pdf', dpi=300)

# Show plot
plt.show()