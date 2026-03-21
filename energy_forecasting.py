# Household Energy Consumption Forecasting
# Dataset: UCI Machine Learning Repository - Individual Household Electric Power Consumption
# Author: Your Name | MNNIT Allahabad

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & INSPECT DATA
# ─────────────────────────────────────────────

def load_data(filepath='data/household_power_consumption.txt'):
    """Load and do initial inspection of the dataset."""
    print("Loading dataset...")
    df = pd.read_csv(
        filepath,
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        dayfirst=True,
        low_memory=False,
        na_values=['?']
    )
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df


# ─────────────────────────────────────────────
# 2. DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_data(df):
    """Clean and engineer features from raw data."""
    print("\n--- Preprocessing ---")

    # Drop rows with missing values
    df.dropna(inplace=True)
    print(f"Shape after dropping nulls: {df.shape}")

    # Convert power columns to numeric
    power_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in power_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    # Feature Engineering from datetime
    df['hour']       = df['datetime'].dt.hour
    df['day']        = df['datetime'].dt.day
    df['month']      = df['datetime'].dt.month
    df['weekday']    = df['datetime'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # Resample to hourly to reduce size & noise
    df.set_index('datetime', inplace=True)
    df_hourly = df.resample('H').mean().dropna()
    df_hourly.reset_index(inplace=True)

    # Re-extract time features after resampling
    df_hourly['hour']       = df_hourly['datetime'].dt.hour
    df_hourly['day']        = df_hourly['datetime'].dt.day
    df_hourly['month']      = df_hourly['datetime'].dt.month
    df_hourly['weekday']    = df_hourly['datetime'].dt.weekday
    df_hourly['is_weekend'] = (df_hourly['weekday'] >= 5).astype(int)

    print(f"Shape after hourly resampling: {df_hourly.shape}")
    return df_hourly


# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

def perform_eda(df):
    """Generate EDA plots."""
    print("\n--- EDA ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Household Energy Consumption - EDA', fontsize=16, fontweight='bold')

    # Average consumption by hour
    hourly_avg = df.groupby('hour')['Global_active_power'].mean()
    axes[0, 0].plot(hourly_avg.index, hourly_avg.values, color='steelblue', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Average Power Consumption by Hour')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Avg Active Power (kW)')
    axes[0, 0].grid(alpha=0.3)

    # Average consumption by month
    monthly_avg = df.groupby('month')['Global_active_power'].mean()
    axes[0, 1].bar(monthly_avg.index, monthly_avg.values, color='coral', edgecolor='black', alpha=0.8)
    axes[0, 1].set_title('Average Power Consumption by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Avg Active Power (kW)')
    axes[0, 1].grid(alpha=0.3, axis='y')

    # Weekday vs Weekend
    df.groupby('is_weekend')['Global_active_power'].mean().plot(
        kind='bar', ax=axes[1, 0], color=['steelblue', 'coral'], edgecolor='black'
    )
    axes[1, 0].set_title('Weekday vs Weekend Consumption')
    axes[1, 0].set_xticklabels(['Weekday', 'Weekend'], rotation=0)
    axes[1, 0].set_ylabel('Avg Active Power (kW)')
    axes[1, 0].grid(alpha=0.3, axis='y')

    # Correlation heatmap
    corr_cols = ['Global_active_power', 'Voltage', 'Global_intensity',
                 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'hour', 'month']
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1], linewidths=0.5)
    axes[1, 1].set_title('Feature Correlation Heatmap')

    plt.tight_layout()
    plt.savefig('outputs/eda_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("EDA plot saved to outputs/eda_plots.png")


# ─────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_and_evaluate(df):
    """Train multiple models and compare performance."""
    print("\n--- Model Training ---")

    features = ['hour', 'day', 'month', 'weekday', 'is_weekend',
                'Voltage', 'Global_intensity', 'Sub_metering_1',
                'Sub_metering_2', 'Sub_metering_3']
    target = 'Global_active_power'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'Linear Regression':       LinearRegression(),
        'Random Forest':           RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting':       GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'model': model, 'y_pred': y_pred}
        print(f"\n{name}:")
        print(f"  MAE  = {mae:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  R²   = {r2:.4f}")

    return results, y_test, scaler, features


# ─────────────────────────────────────────────
# 5. VISUALIZE RESULTS
# ─────────────────────────────────────────────

def plot_results(results, y_test):
    """Plot actual vs predicted and model comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Evaluation Results', fontsize=14, fontweight='bold')

    # Actual vs Predicted (best model = Random Forest)
    best_name = max(results, key=lambda x: results[x]['R2'])
    y_pred_best = results[best_name]['y_pred']
    sample = min(200, len(y_test))

    axes[0].plot(range(sample), list(y_test)[:sample], label='Actual', color='steelblue', linewidth=1.5)
    axes[0].plot(range(sample), y_pred_best[:sample], label=f'Predicted ({best_name})', color='coral', linewidth=1.5, linestyle='--')
    axes[0].set_title('Actual vs Predicted (first 200 samples)')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Global Active Power (kW)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Model comparison bar chart
    model_names = list(results.keys())
    r2_scores   = [results[m]['R2'] for m in model_names]
    colors = ['steelblue', 'coral', 'seagreen']
    bars = axes[1].bar(model_names, r2_scores, color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_title('R² Score Comparison')
    axes[1].set_ylabel('R² Score')
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3, axis='y')
    for bar, score in zip(bars, r2_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{score:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/model_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Results plot saved to outputs/model_results.png")


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_feature_importance(results, features):
    """Plot feature importance from Random Forest."""
    rf_model = results['Random Forest']['model']
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    importances.plot(kind='barh', color='steelblue', edgecolor='black', alpha=0.85)
    plt.title('Feature Importance - Random Forest', fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Feature importance plot saved to outputs/feature_importance.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)

    df = load_data()
    df = preprocess_data(df)
    perform_eda(df)
    results, y_test, scaler, features = train_and_evaluate(df)
    plot_results(results, y_test)
    plot_feature_importance(results, features)

    print("\n✅ Project complete! Check the outputs/ folder for all plots.")
