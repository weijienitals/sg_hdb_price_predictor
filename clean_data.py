import pandas as pd
import numpy as np

print("=" * 70)
print("HDB DATA CLEANING SCRIPT")
print("=" * 70)
print("\nThis script will:")
print("1. Remove invalid flat types (non-HDB types like 'TERRACE')")
print("2. Remove unrealistic entries")
print("3. Create a reference file of valid combinations")
print("4. Save cleaned data for retraining")
print("\n" + "=" * 70)

# Load original data
print("\n[STEP 1] Loading original data...")
try:
    df = pd.read_csv('resale-flat-prices.csv')
    print(f"✅ Loaded {len(df):,} rows")
except FileNotFoundError:
    print("❌ ERROR: Cannot find 'resale-flat-prices.csv'")
    print("   Make sure this file is in the same folder as this script!")
    exit(1)

original_count = len(df)

# Define valid HDB flat types (official HDB types only)
print("\n[STEP 2] Checking for invalid flat types...")
VALID_HDB_TYPES = [
    '1 ROOM', 
    '2 ROOM', 
    '3 ROOM', 
    '4 ROOM', 
    '5 ROOM', 
    'EXECUTIVE', 
    'MULTI-GENERATION',
    'MULTI GENERATION'  # Handle both versions
]

# Check what flat types exist
print(f"\nFlat types found in data:")
for ft in sorted(df['flat_type'].unique()):
    count = len(df[df['flat_type'] == ft])
    is_valid = "✅" if ft in VALID_HDB_TYPES else "❌"
    print(f"  {is_valid} {ft}: {count:,} rows")

# Remove invalid flat types
invalid_types = df[~df['flat_type'].isin(VALID_HDB_TYPES)]
if len(invalid_types) > 0:
    print(f"\n⚠️  Removing {len(invalid_types):,} rows with invalid flat types")
    print(f"   Invalid types: {invalid_types['flat_type'].unique().tolist()}")
    df = df[df['flat_type'].isin(VALID_HDB_TYPES)]
else:
    print("✅ All flat types are valid HDB types")

# Check for unrealistic storey ranges
print("\n[STEP 3] Checking storey ranges...")
df['temp_max_floor'] = df['storey_range'].str.extract('TO (\d+)').astype(int)
max_floor_overall = df['temp_max_floor'].max()
print(f"Maximum floor in dataset: {max_floor_overall}")

# HDB rarely goes above 50 floors, flag if found
unrealistic = df[df['temp_max_floor'] > 50]
if len(unrealistic) > 0:
    print(f"⚠️  Found {len(unrealistic)} flats above 50 floors (unusual but keeping)")
    print(f"   Towns: {unrealistic['town'].unique().tolist()}")
else:
    print("✅ All storey ranges look realistic")

df = df.drop('temp_max_floor', axis=1)

# Check floor areas
print("\n[STEP 4] Checking floor areas...")
print("\nFloor area ranges by flat type:")
for flat_type in sorted(df['flat_type'].unique()):
    subset = df[df['flat_type'] == flat_type]
    print(f"  {flat_type:20s}: {subset['floor_area_sqm'].min():6.1f} - {subset['floor_area_sqm'].max():6.1f} sqm (median: {subset['floor_area_sqm'].median():6.1f})")

# Remove extreme outliers (likely data errors)
print("\nRemoving extreme floor area outliers...")
for flat_type in df['flat_type'].unique():
    mask = df['flat_type'] == flat_type
    q1 = df.loc[mask, 'floor_area_sqm'].quantile(0.001)  # Bottom 0.1%
    q99 = df.loc[mask, 'floor_area_sqm'].quantile(0.999)  # Top 0.1%
    
    outliers = df[mask & ((df['floor_area_sqm'] < q1) | (df['floor_area_sqm'] > q99))]
    if len(outliers) > 0:
        print(f"  Removing {len(outliers)} extreme outliers from {flat_type}")
        df = df[~mask | ((df['floor_area_sqm'] >= q1) & (df['floor_area_sqm'] <= q99))]

# Check lease dates
print("\n[STEP 5] Checking lease commence dates...")
min_lease = df['lease_commence_date'].min()
max_lease = df['lease_commence_date'].max()
print(f"Lease date range: {min_lease} - {max_lease}")

unrealistic_lease = df[(df['lease_commence_date'] < 1960) | (df['lease_commence_date'] > 2025)]
if len(unrealistic_lease) > 0:
    print(f"⚠️  Removing {len(unrealistic_lease)} properties with unrealistic lease dates")
    df = df[(df['lease_commence_date'] >= 1960) & (df['lease_commence_date'] <= 2025)]
else:
    print("✅ All lease dates look good")

# Remove duplicates
print("\n[STEP 6] Checking for duplicates...")
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f"⚠️  Removing {duplicates} duplicate rows")
    df = df.drop_duplicates()
else:
    print("✅ No duplicates found")

# Check price outliers
print("\n[STEP 7] Checking prices...")
print(f"Price range: ${df['resale_price'].min():,.0f} - ${df['resale_price'].max():,.0f}")
print(f"Mean: ${df['resale_price'].mean():,.0f}, Median: ${df['resale_price'].median():,.0f}")

# Flag extreme price outliers (but keep them as they might be real)
mean_price = df['resale_price'].mean()
std_price = df['resale_price'].std()
outliers = df[(df['resale_price'] < mean_price - 4*std_price) | 
               (df['resale_price'] > mean_price + 4*std_price)]
if len(outliers) > 0:
    print(f"⚠️  Found {len(outliers)} extreme price outliers (>4 std dev)")
    print(f"   These might be legitimate luxury/subsidized units, keeping them")

# SAVE CLEANED DATA
print("\n" + "=" * 70)
print("[STEP 8] SAVING CLEANED DATA")
print("=" * 70)

df.to_csv('resale-flat-prices-cleaned.csv', index=False)
print(f"\n✅ SAVED: resale-flat-prices-cleaned.csv")
print(f"   Original rows: {original_count:,}")
print(f"   Cleaned rows:  {len(df):,}")
print(f"   Removed:       {original_count - len(df):,} ({(original_count - len(df))/original_count*100:.2f}%)")

# CREATE VALID COMBINATIONS REFERENCE
print("\n[STEP 9] Creating valid combinations reference...")

# Town + Flat Type combinations
town_type = df.groupby(['town', 'flat_type']).size().reset_index(name='count')
town_type.to_csv('valid_town_flattype.csv', index=False)
print(f"✅ SAVED: valid_town_flattype.csv ({len(town_type)} combinations)")

# Town + Flat Type + Model combinations
town_type_model = df.groupby(['town', 'flat_type', 'flat_model']).size().reset_index(name='count')
town_type_model.to_csv('valid_town_flattype_model.csv', index=False)
print(f"✅ SAVED: valid_town_flattype_model.csv ({len(town_type_model)} combinations)")

# Storey ranges by town
town_storey = df.groupby(['town', 'storey_range']).size().reset_index(name='count')
town_storey.to_csv('valid_town_storey.csv', index=False)
print(f"✅ SAVED: valid_town_storey.csv ({len(town_storey)} combinations)")

# Summary statistics by town
print("\n[STEP 10] Creating town statistics...")
town_stats = df.groupby('town').agg({
    'resale_price': ['count', 'min', 'max', 'median'],
    'floor_area_sqm': ['min', 'max', 'median'],
    'lease_commence_date': ['min', 'max']
}).round(0)
town_stats.columns = ['_'.join(col).strip() for col in town_stats.columns.values]
town_stats.to_csv('town_statistics.csv')
print(f"✅ SAVED: town_statistics.csv")

# SUMMARY
print("\n" + "=" * 70)
print("✨ DATA CLEANING COMPLETE!")
print("=" * 70)
print("\nFiles created:")
print("  1. resale-flat-prices-cleaned.csv      <- Use this for training!")
print("  2. valid_town_flattype.csv              <- Reference for valid combos")
print("  3. valid_town_flattype_model.csv        <- Reference for valid combos")
print("  4. valid_town_storey.csv                <- Reference for valid combos")
print("  5. town_statistics.csv                  <- Stats by town")

print("\n📋 NEXT STEPS:")
print("  1. Run: python train_model_updated.py")
print("  2. Then run: streamlit run app_updated.py")
print("\n" + "=" * 70)