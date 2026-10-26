import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. GENERATE DATA (Simulating the Saturation Hypothesis)
# ---------------------------------------------------------
np.random.seed(42)
years = list(range(2000, 2024))

# India: GDP rising, Soda rising (developing economy trend)
ind_gdp = np.linspace(400, 2500, 24) * (1 + np.random.normal(0, 0.05, 24))
ind_soda = np.linspace(5, 25, 24) * (1 + np.random.normal(0, 0.05, 24))

# USA: GDP rising, Soda declining (saturation in a mature economy)
usa_gdp = np.linspace(36000, 65000, 24) * (1 + np.random.normal(0, 0.03, 24))
usa_soda = np.linspace(170, 120, 24) * (1 + np.random.normal(0, 0.02, 24))

# UK: GDP rising, Soda steady then sharp drop in 2018 (tax implementation)
gbr_gdp = np.linspace(30000, 48000, 24) * (1 + np.random.normal(0, 0.03, 24))
gbr_soda = [95 + np.random.normal(0, 2) for _ in range(18)] + [76 + np.random.normal(0, 2) for _ in range(6)]

# Combine into a single DataFrame
df = pd.DataFrame({
    'country': ['IND']*24 + ['USA']*24 + ['GBR']*24,
    'year': years * 3,
    'gdp_pc': np.concatenate([ind_gdp, usa_gdp, gbr_gdp]),
    'soda_litres_pc': np.concatenate([ind_soda, usa_soda, gbr_soda])
})

# Add synthetic population and urbanization for the heatmap
df['pop'] = np.random.uniform(10, 100, len(df))
df['urban'] = np.random.uniform(30, 80, len(df))

# ---------------------------------------------------------
# 2. PREPROCESSING & FEATURE ENGINEERING
# ---------------------------------------------------------
# Log transformations for regression
df['log_gdp'] = np.log(df['gdp_pc'])
df['log_soda'] = np.log(df['soda_litres_pc'])

# Calculate Indices (Base 2000 = 100) and Year-over-Year Growth
for c in df['country'].unique():
    mask = df['country'] == c
    # Indexing
    df.loc[mask, 'gdp_pc_idx'] = (df.loc[mask, 'gdp_pc'] / df.loc[mask, 'gdp_pc'].iloc[0]) * 100
    df.loc[mask, 'soda_idx'] = (df.loc[mask, 'soda_litres_pc'] / df.loc[mask, 'soda_litres_pc'].iloc[0]) * 100
    # YoY Growth
    df.loc[mask, 'gdp_g'] = df.loc[mask, 'gdp_pc'].pct_change() * 100
    df.loc[mask, 'soda_g'] = df.loc[mask, 'soda_litres_pc'].pct_change() * 100

# Set visualization theme
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# ---------------------------------------------------------

# --- Figure 1: Indexed Growth ---
plt.figure(figsize=(6, 4))
sns.lineplot(data=df, x='year', y='gdp_pc_idx', hue='country', linestyle='--', legend=False)
sns.lineplot(data=df, x='year', y='soda_idx', hue='country', linewidth=2)
plt.title("Fig 1: Indexed Growth (Base 2000=100)\nGDP (Dashed) vs Soda (Solid)")
plt.ylabel("Index (2000=100)")
plt.tight_layout()
plt.show()

# --- Figure 2: Regression (Log GDP vs Log Soda) ---
plt.figure(figsize=(6, 4))
for country in df['country'].unique():
    sub = df[df['country'] == country]
    sns.regplot(data=sub, x='log_gdp', y='log_soda', scatter=True, label=country)
plt.title("Fig 2: Regression (Log GDP vs Log Soda)")
plt.xlabel("Log(GDP per capita)")
plt.ylabel("Log(Soda Consumption)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Figure 3: Correlation Heatmap ---
plt.figure(figsize=(5, 4))
# Select only the numeric columns for correlation
corr_matrix = df[['gdp_pc', 'soda_litres_pc', 'pop', 'urban']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Fig 3: Correlation Heatmap")
plt.tight_layout()
plt.show()

# --- Figure 4: Year-over-Year Growth ---
plt.figure(figsize=(6, 4))
sns.lineplot(data=df, x='year', y='soda_g', hue='country', marker='o')
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.title("Fig 4: YoY Soda Consumption Growth (%)")
plt.ylabel("Growth Rate (%)")
plt.tight_layout()
plt.show()