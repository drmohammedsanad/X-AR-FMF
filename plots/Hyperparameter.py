import matplotlib.pyplot as plt

# Data
beta = [0.00, 0.01, 0.03, 0.05, 0.10]
rmse = [2.6067, 2.6118, 2.6194, 2.6308, 2.6547]
coverage = [0.0000, 0.3125, 0.5872, 0.8760, 0.9413]

# Create figure
fig, ax1 = plt.subplots(figsize=(7,5))

# RMSE axis
color = 'tab:blue'
ax1.set_xlabel(r'Fairness Parameter $\beta$', fontsize=12)
ax1.set_ylabel('RMSE', color=color, fontsize=12)
ax1.plot(beta, rmse, marker='o', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

# Coverage axis
ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Coverage', color=color, fontsize=12)
ax2.plot(beta, coverage, marker='s', linestyle='--', color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

# Title
plt.title('Effect of Fairness Parameter on Accuracy and Explainability')

# Grid
ax1.grid(True, linestyle='--', alpha=0.5)

# Layout
fig.tight_layout()

# Save
#plt.savefig('beta_sensitivity.pdf', dpi=300)
plt.savefig('beta_sensitivity.png', dpi=300)

plt.show()