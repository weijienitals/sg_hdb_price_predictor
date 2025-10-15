import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('resale-flat-prices.csv')

# Basic exploration
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe())

# Check unique values for categorical columns
print("\nUnique towns:", df['town'].nunique())
print("\nUnique flat types:", df['flat_type'].unique())
print("\nUnique flat models:", df['flat_model'].unique())

# Visualize price distribution
plt.figure(figsize=(10, 6))
plt.hist(df['resale_price'], bins=50, edgecolor='black')
plt.xlabel('Resale Price')
plt.ylabel('Frequency')
plt.title('Distribution of HDB Resale Prices')
plt.savefig('price_distribution.png')
plt.close()

# Price by town
plt.figure(figsize=(15, 8))
df.groupby('town')['resale_price'].median().sort_values().plot(kind='barh')
plt.xlabel('Median Resale Price')
plt.title('Median HDB Resale Price by Town')
plt.tight_layout()
plt.savefig('price_by_town.png')
plt.close()

print("\nExploration complete! Check the generated PNG files.")