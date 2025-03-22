import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# Set random seed
np.random.seed(42)

# Generate categorical data for two groups
group_A = np.random.choice(["Engineer", "Doctor", "Teacher"], size=100, p=[0.5, 0.3, 0.2])
group_B = np.random.choice(["Engineer", "Doctor", "Teacher"], size=100, p=[0.3, 0.4, 0.3])  # Different distribution

# Convert to DataFrame
df_A = pd.DataFrame(group_A, columns=["Job"])
df_B = pd.DataFrame(group_B, columns=["Job"])

# One-hot encoding
df_A_encoded = pd.get_dummies(df_A, columns=["Job"])
df_B_encoded = pd.get_dummies(df_B, columns=["Job"])

# Compute category proportions
job_counts_A = df_A["Job"].value_counts(normalize=True)
job_counts_B = df_B["Job"].value_counts(normalize=True)

# Perform KS Test for each category
ks_results = []
for category in df_A_encoded.columns:
    stat, p_value = ks_2samp(df_A_encoded[category], df_B_encoded[category])
    ks_results.append((category, stat, p_value))

# Convert results to DataFrame
ks_df = pd.DataFrame(ks_results, columns=["Category", "KS Statistic", "p-value"])

# Plot the distributions
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(x=job_counts_A.index, y=job_counts_A.values, ax=ax[0], hue = job_counts_A.index, palette="Blues", legend=False)
ax[0].set_title("Job Distribution in Group A")
ax[0].set_ylabel("Proportion")
ax[0].set_ylim(0, 0.6)

sns.barplot(x=job_counts_B.index, y=job_counts_B.values, ax=ax[1], hue = job_counts_B.index, palette="Oranges",  legend=False)

ax[1].set_title("Job Distribution in Group B")
ax[1].set_ylabel("Proportion")
ax[1].set_ylim(0, 0.6)

plt.show()

# Display KS test results
print(ks_df)
