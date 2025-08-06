# E-commerce Supply Chain Optimization Project
# Author: Your Team
# Date: August 1, 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data(file_path, sample_size=20000):
    """
    Load the supply chain dataset and perform initial exploration
    """
    print("Loading supply chain data...")
    
    try:
        # Load the dataset with proper encoding handling
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:  
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp1252')
        
        print(f"✓ Dataset loaded successfully!")
        print(f"✓ Shape before sampling: {df.shape}")
        
        # Sample the dataset if it exceeds the sample size
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"✓ Dataset sampled to {sample_size} rows")
        
        # Basic information about the dataset
        print("\n" + "="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        print(f"Number of rows: {df.shape[0]:,}")
        print(f"Number of columns: {df.shape[1]}")
        
        print("\nColumn Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\nData Types:")
        print(df.dtypes)
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nBasic Statistics:")
        print(df.describe())
        
        print("\nMissing Values:")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: Dataset file not found!")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis")
        print("And place it in the current directory.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def clean_data(df):
    """
    Clean and preprocess the supply chain data
    """
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)
    
    if df is None:
        return None
    
    # Make a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Convert date columns to datetime
    date_columns = []
    for col in df_clean.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df_clean[col] = pd.to_datetime(df_clean[col])
                date_columns.append(col)
                print(f"✓ Converted {col} to datetime")
            except:
                print(f"⚠ Could not convert {col} to datetime")
    
    # Handle missing values
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if df_clean[col].isnull().sum() > 0:
            # Fill with median for numeric columns
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"✓ Filled missing values in {col} with median: {median_val}")
    
    # Fill categorical missing values with mode
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col not in date_columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col].fillna(mode_val[0], inplace=True)
                print(f"✓ Filled missing values in {col} with mode: {mode_val[0]}")
    
    print(f"✓ Data cleaning completed!")
    print(f"✓ Final shape: {df_clean.shape}")
    
    return df_clean

def exploratory_data_analysis(df):
    """
    Perform exploratory data analysis on the supply chain data
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)

    if df is None:
        return

    # Create visualizations directory
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    # 1. Sales trends over time
    plt.figure(figsize=(15, 10))

    # Find date and sales columns
    date_cols = []
    sales_cols = []

    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            if df[col].dtype == 'datetime64[ns]':
                date_cols.append(col)
        if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'price', 'total']):
            if df[col].dtype in ['int64', 'float64']:
                sales_cols.append(col)

    print(f"Found date columns: {date_cols}")
    print(f"Found sales-related columns: {sales_cols}")

    # 2. Top products analysis
    product_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['product', 'item', 'category']):
            product_cols.append(col)

    print(f"Found product-related columns: {product_cols}")

    # 3. Geographic analysis
    geo_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['country', 'city', 'region', 'state']):
            geo_cols.append(col)

    print(f"Found geographic columns: {geo_cols}")

    # Generate basic plots
    plt.figure(figsize=(15, 5))

    # Plot 1: Distribution of numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plt.subplot(1, 3, 1)
        df[numeric_cols[0]].hist(bins=30, alpha=0.7, color='salmon')
        plt.title(f'Distribution of {numeric_cols[0]}', fontsize=12)
        plt.xlabel(numeric_cols[0], fontsize=10)
        plt.ylabel('Frequency', fontsize=10)

    # Plot 2: Top categories if available
    if len(product_cols) > 0:
        plt.subplot(1, 3, 2)
        top_products = df[product_cols[0]].value_counts().head(10)
        top_products.plot(kind='bar', color='salmon')
        plt.title(f'Top 10 {product_cols[0]}', fontsize=12)
        plt.xticks(rotation=45, fontsize=9)
        plt.yticks(fontsize=9)

    # Plot 3: Correlation matrix
    if len(numeric_cols) > 1:
        plt.figure(figsize=(20, 15))  # Increase figure size
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', annot_kws={"fontsize": 10})
        plt.title('Correlation Matrix', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)  # Rotate labels
        plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig('visualizations/basic_eda.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Basic EDA completed! Visualizations saved to 'visualizations/' folder")

def demand_forecasting_setup(df):
    """
    Set up demand forecasting analysis
    """
    print("\n" + "="*50)
    print("DEMAND FORECASTING SETUP")
    print("="*50)
    
    if df is None:
        return
    
    # This is a template for demand forecasting
    # We'll need to identify the right columns for time series analysis
    
    print("Setting up demand forecasting models...")
    print("📊 Time series analysis requires:")
    print("   - Date column (for time axis)")
    print("   - Demand/Sales column (for forecasting)")
    print("   - Optional: Product/Category grouping")
    
    # Find suitable columns for forecasting
    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nAvailable date columns: {date_cols}")
    print(f"Available numeric columns for forecasting: {numeric_cols[:10]}...")  # Show first 10
    
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        print("✓ Data is suitable for time series forecasting!")
        print("📈 Next steps: Implement ARIMA, Prophet, or LSTM models")
    else:
        print("⚠ Need to identify appropriate date and numeric columns for forecasting")

def main():
    """
    Main function to run the supply chain optimization analysis
    """
    print("🚀 STARTING E-COMMERCE SUPPLY CHAIN OPTIMIZATION PROJECT")
    print("=" * 60)
    
    # Step 1: Load and explore data
    # Using DataCo dataset (the actual Kaggle dataset)
    data_file = 'DataCoSupplyChainDataset.csv'  # Updated to use the correct file
    
    df = load_and_explore_data(data_file)
    
    if df is not None:
        # Step 2: Clean the data
        df_clean = clean_data(df)
        
        # Step 3: Perform EDA
        exploratory_data_analysis(df_clean)
        
        # Step 4: Set up demand forecasting
        demand_forecasting_setup(df_clean)
        
        print("\n" + "="*60)
        print("🎯 PROJECT SETUP COMPLETED!")
        print("📂 Next steps:")
        print("   1. Download the dataset and update the filename")
        print("   2. Run this script to explore your data")
        print("   3. Implement specific forecasting models")
        print("   4. Build inventory optimization algorithms")
        print("   5. Create performance dashboards")
    else:
        print("\n📥 Please download the dataset first and update the filename in the script.")

if __name__ == "__main__":
    main()
