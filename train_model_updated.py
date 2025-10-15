import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("=" * 60)
print("HDB RESALE PRICE PREDICTION - MODEL TRAINING PIPELINE")
print("Using CLEANED data with validation")
print("=" * 60)

# Load CLEANED data
try:
    df = pd.read_csv('resale-flat-prices-cleaned.csv')
    print(f"\n✅ Loaded CLEANED data: {df.shape}")
except FileNotFoundError:
    print("\n❌ ERROR: resale-flat-prices-cleaned.csv not found!")
    print("   Run this first: python clean_data.py")
    exit(1)

# Feature Engineering
print("\n[1/6] Feature Engineering...")
df['year'] = pd.to_datetime(df['month']).dt.year
df['month_num'] = pd.to_datetime(df['month']).dt.month
df['remaining_lease_years'] = df['remaining_lease'].str.extract('(\d+)').astype(int)
df['flat_age'] = df['year'] - df['lease_commence_date']
df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']

# Select features
features_to_use = [
    'town', 'flat_type', 'flat_model', 'floor_area_sqm',
    'lease_commence_date', 'year', 'month_num', 
    'storey_range', 'remaining_lease_years', 'flat_age'
]

X = df[features_to_use].copy()
y = df['resale_price']

# Encode categorical variables
print("[2/6] Encoding categorical variables...")
label_encoders = {}
categorical_cols = ['town', 'flat_type', 'flat_model', 'storey_range']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train):,}")
print(f"Test set size: {len(X_test):,}")

# BASELINE MODEL
print("\n[3/6] Training baseline model...")
baseline_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)

print(f"\n📊 BASELINE MODEL PERFORMANCE:")
print(f"   MAE:  ${baseline_mae:,.2f}")
print(f"   RMSE: ${baseline_rmse:,.2f}")
print(f"   R²:   {baseline_r2:.4f}")

# HYPERPARAMETER TUNING
print("\n[4/6] Hyperparameter tuning (this may take a few minutes)...")

param_distributions = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

random_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=20,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print(f"\n✅ Best parameters found:")
for param, value in random_search.best_params_.items():
    print(f"   {param}: {value}")

# Train final model
print("\n[5/6] Training optimized model...")
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100
rmse_improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
r2_improvement = ((r2 - baseline_r2) / baseline_r2) * 100

print("\n" + "=" * 60)
print("📈 FINAL MODEL PERFORMANCE")
print("=" * 60)
print(f"\nMetric                    Baseline        Optimized       Improvement")
print("-" * 60)
print(f"MAE                      ${baseline_mae:>10,.0f}    ${mae:>10,.0f}    {mae_improvement:>6.2f}%")
print(f"RMSE                     ${baseline_rmse:>10,.0f}    ${rmse:>10,.0f}    {rmse_improvement:>6.2f}%")
print(f"R² Score                  {baseline_r2:>10.4f}     {r2:>10.4f}    {r2_improvement:>6.2f}%")
print(f"MAPE                          -           {mape:>10.2f}%        -")
print("\n" + "=" * 60)

# Cross-validation
cv_scores = cross_val_score(
    best_model, X_train, y_train, 
    cv=5, 
    scoring='neg_mean_absolute_error'
)
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"\n🔄 Cross-Validation (5-fold):")
print(f"   Mean MAE: ${cv_mae:,.2f} (+/- ${cv_std:,.2f})")

# Visualizations
print("\n📊 Generating evaluation plots...")

# 1. Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)', fontsize=12)
plt.ylabel('Predicted Price ($)', fontsize=12)
plt.title('Predicted vs Actual Prices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=300)
plt.close()

# 2. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5, s=10)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price ($)', fontsize=12)
plt.ylabel('Residuals ($)', fontsize=12)
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300)
plt.close()

# 3. Error Distribution
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300)
plt.close()

# 4. Feature Importance
feature_importance = pd.DataFrame({
    'feature': features_to_use,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

print("\n✅ Visualizations saved:")
print("   - predicted_vs_actual.png")
print("   - residual_plot.png")
print("   - error_distribution.png")
print("   - feature_importance.png")

# Error Analysis
print("\n" + "=" * 60)
print("🔍 ERROR ANALYSIS")
print("=" * 60)

percentage_errors = np.abs((y_test - y_pred) / y_test) * 100

print(f"\nPercentage Error Statistics:")
print(f"   Mean:   {percentage_errors.mean():.2f}%")
print(f"   Median: {np.median(percentage_errors):.2f}%")
print(f"   Std:    {percentage_errors.std():.2f}%")
print(f"\n   Within 5%:  {(percentage_errors <= 5).sum() / len(percentage_errors) * 100:.2f}% of predictions")
print(f"   Within 10%: {(percentage_errors <= 10).sum() / len(percentage_errors) * 100:.2f}% of predictions")
print(f"   Within 15%: {(percentage_errors <= 15).sum() / len(percentage_errors) * 100:.2f}% of predictions")

# SAVE MODELS
print("\n" + "=" * 60)
print("💾 Saving models and artifacts...")
print("=" * 60)

with open('hdb_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('baseline_model.pkl', 'wb') as f:
    pickle.dump(baseline_model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(features_to_use, f)

# Save metrics
metrics = {
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'mape': mape,
    'cv_mae': cv_mae,
    'cv_std': cv_std,
    'baseline_mae': baseline_mae,
    'baseline_rmse': baseline_rmse,
    'baseline_r2': baseline_r2,
    'improvement_mae': mae_improvement,
    'improvement_rmse': rmse_improvement,
    'improvement_r2': r2_improvement
}

with open('model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("\n✅ Files saved:")
print("   - hdb_model.pkl")
print("   - baseline_model.pkl")
print("   - label_encoders.pkl")
print("   - feature_names.pkl")
print("   - model_metrics.pkl")

print("\n" + "=" * 60)
print("✨ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\n📋 NEXT STEP:")
print("   Run: streamlit run app_updated.py")
print("\n" + "=" * 60)