# Advanced Supply Chain Analysis - Demand Forecasting & Inventory Optimization
# This script implements specific forecasting models and optimization algorithms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import os

def load_and_prepare_data():
    """Load and prepare data for advanced analysis"""
    print("📊 Loading DataCo Supply Chain dataset for advanced analysis...")

    # Load the DataCo dataset with proper encoding
    try:
        df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='cp1252')

    print(f"✓ Loaded {len(df)} records with {len(df.columns)} original columns")

    # Limit the dataset to 10,000 rows
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
        print(f"✓ Dataset sampled to 10,000 rows")

    # Map DataCo columns to standardized format for analysis
    column_mapping = {
        'order date (DateOrders)': 'Order_Date',
        'shipping date (DateOrders)': 'Ship_Date', 
        'Category Name': 'Product_Category',
        'Product Name': 'Product_Name',
        'Customer Segment': 'Customer_Segment',
        'Shipping Mode': 'Shipping_Mode',
        'Order Item Quantity': 'Quantity_Ordered',
        'Product Price': 'Unit_Price',
        'Sales': 'Total_Sales',
        'Order Profit Per Order': 'Profit',
        'Days for shipping (real)': 'Delivery_Days',
        'Customer City': 'City',
        'Customer Country': 'Country',
        'Order Region': 'Region',
        'Late_delivery_risk': 'Late_Delivery_Risk',
        'Order Item Total': 'Order_Total',
        'Order Item Profit Ratio': 'Profit_Ratio'
    }

    # Rename available columns
    available_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=available_mappings)

    # Create derived columns needed for analysis
    if 'Total_Sales' not in df.columns and 'Sales' in df.columns:
        df['Total_Sales'] = df['Sales']

    if 'Profit' not in df.columns and 'Benefit per order' in df.columns:
        df['Profit'] = df['Benefit per order']

    # Create missing columns with calculated/estimated values
    if 'Unit_Cost' not in df.columns:
        df['Unit_Cost'] = df['Unit_Price'] * 0.65 if 'Unit_Price' in df.columns else df['Total_Sales'] * 0.65

    if 'Total_Cost' not in df.columns:
        df['Total_Cost'] = df['Unit_Cost'] * df['Quantity_Ordered'] if 'Quantity_Ordered' in df.columns else df['Total_Sales'] * 0.65

    # Calculate delivery metrics
    if 'On_Time_Delivery' not in df.columns:
        if 'Late_Delivery_Risk' in df.columns:
            df['On_Time_Delivery'] = 1 - df['Late_Delivery_Risk']
        else:
            df['On_Time_Delivery'] = np.random.uniform(0.7, 0.95, len(df))

    # Estimate customer satisfaction based on delivery performance
    if 'Customer_Satisfaction' not in df.columns:
        if 'On_Time_Delivery' in df.columns:
            # Higher on-time delivery correlates with higher satisfaction
            base_satisfaction = 2.5 + (df['On_Time_Delivery'] * 2)
            noise = np.random.normal(0, 0.3, len(df))
            df['Customer_Satisfaction'] = np.clip(base_satisfaction + noise, 1, 5)
        else:
            df['Customer_Satisfaction'] = np.random.uniform(2, 4.5, len(df))

    # Generate inventory-related columns for optimization analysis
    if 'Inventory_Level' not in df.columns:
        # Estimate inventory based on order quantity patterns
        np.random.seed(42)  # For reproducible results
        df['Inventory_Level'] = df['Quantity_Ordered'] * np.random.uniform(5, 20, len(df))

    if 'Reorder_Point' not in df.columns:
        df['Reorder_Point'] = df['Inventory_Level'] * np.random.uniform(0.15, 0.25, len(df))

    if 'Stock_Status' not in df.columns:
        # Determine stock status based on inventory vs reorder point
        conditions = [
            df['Inventory_Level'] <= df['Reorder_Point'],
            df['Inventory_Level'] <= df['Reorder_Point'] * 2,
            df['Inventory_Level'] > df['Reorder_Point'] * 2
        ]
        choices = ['Out of Stock', 'Low Stock', 'In Stock']
        df['Stock_Status'] = np.select(conditions, choices, default='In Stock')

    # Generate operational columns
    if 'Order_Priority' not in df.columns:
        df['Order_Priority'] = np.random.choice(['Low', 'Medium', 'High'], len(df), p=[0.4, 0.4, 0.2])

    if 'Supplier' not in df.columns:
        df['Supplier'] = 'Supplier_' + np.random.choice(['A', 'B', 'C', 'D', 'E'], len(df))

    if 'Warehouse' not in df.columns:
        df['Warehouse'] = 'Warehouse_' + np.random.choice(['North', 'South', 'East', 'West', 'Central'], len(df))

    # Convert and create date columns
    date_cols = ['Order_Date', 'Ship_Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Create delivery date if not exists
    if 'Delivery_Date' not in df.columns and 'Order_Date' in df.columns and 'Delivery_Days' in df.columns:
        df['Delivery_Date'] = df['Order_Date'] + pd.to_timedelta(df['Delivery_Days'], unit='days')
    elif 'Delivery_Date' not in df.columns and 'Order_Date' in df.columns:
        df['Delivery_Date'] = df['Order_Date'] + pd.to_timedelta(np.random.randint(1, 20, len(df)), unit='days')

    # Create additional time-based features
    df['Order_Year'] = df['Order_Date'].dt.year
    df['Order_Month'] = df['Order_Date'].dt.month
    df['Order_Quarter'] = df['Order_Date'].dt.quarter
    df['Order_DayOfWeek'] = df['Order_Date'].dt.dayofweek
    df['Order_WeekOfYear'] = df['Order_Date'].dt.isocalendar().week

    # Calculate profit margin
    df['Profit_Margin'] = (df['Profit'] / df['Total_Sales']) * 100

    # Stock efficiency metrics
    df['Stock_Efficiency'] = df['Quantity_Ordered'] / df['Inventory_Level']
    df['Days_To_Reorder'] = np.where(df['Inventory_Level'] <= df['Reorder_Point'], 0, 
                                    (df['Inventory_Level'] - df['Reorder_Point']) / df['Quantity_Ordered'])

    print(f"✓ Data prepared with {len(df)} records and {len(df.columns)} features")
    return df

def time_series_analysis(df):
    """Perform time series analysis for demand forecasting"""
    print("\n" + "="*60)
    print("� TIME SERIES ANALYSIS")
    print("="*60)
    
    # 1. Time Series Analysis by Product Category
    daily_demand = df.groupby(['Order_Date', 'Product_Category']).agg({
        'Quantity_Ordered': 'sum',
        'Total_Sales': 'sum'
    }).reset_index()
    
    # Focus on top 3 categories
    top_categories = df['Product_Category'].value_counts().head(3).index.tolist()
    
    plt.figure(figsize=(20, 12))
    
    for i, category in enumerate(top_categories, 1):
        category_data = daily_demand[daily_demand['Product_Category'] == category].copy()
        category_data = category_data.sort_values('Order_Date')
        category_data.set_index('Order_Date', inplace=True)
        
        # Resample to monthly data for cleaner visualization
        monthly_data = category_data.resample('M').agg({
            'Quantity_Ordered': 'sum',
            'Total_Sales': 'sum'
        })
        
        plt.subplot(3, 3, i)
        plt.plot(monthly_data.index, monthly_data['Quantity_Ordered'], marker='o', color='blue')
        plt.title(f'Monthly Demand - {category}', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Quantity Ordered', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Trend analysis
        plt.subplot(3, 3, i+3)
        if len(monthly_data) >= 24:  # Need at least 2 years for seasonal decomposition
            decomposition = seasonal_decompose(monthly_data['Quantity_Ordered'], model='additive', period=12)
            decomposition.trend.plot(color='red')
        else:
            # Simple trend line for shorter data
            plt.plot(monthly_data.index, monthly_data['Quantity_Ordered'], color='red')
        plt.title(f'Trend - {category}', fontsize=14)
        plt.ylabel('Trend', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Forecast using Simple Moving Average
        plt.subplot(3, 3, i+6)
        window_size = min(3, len(monthly_data) - 1)
        monthly_data['MA_Forecast'] = monthly_data['Quantity_Ordered'].rolling(window=window_size).mean()
        plt.plot(monthly_data.index, monthly_data['Quantity_Ordered'], label='Actual', marker='o', color='blue')
        plt.plot(monthly_data.index, monthly_data['MA_Forecast'], label='MA Forecast', marker='s', color='orange')
        plt.title(f'Moving Average Forecast - {category}', fontsize=14)
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Time series analysis completed")

def ml_forecasting_analysis(df):
    """Perform machine learning forecasting analysis"""
    print("\n" + "="*60)
    print("🤖 MACHINE LEARNING FORECASTING")
    print("="*60)
    
    # 2. Machine Learning Forecasting
    print("\n📈 Building ML forecasting models...")
    
    # Prepare features for ML model
    ml_data = df.copy()
    
    # Create lag features
    ml_data = ml_data.sort_values(['Product_Category', 'Order_Date'])
    
    # Features for prediction
    feature_cols = ['Order_Month', 'Order_Quarter', 'Order_DayOfWeek', 'Order_WeekOfYear',
                   'Unit_Price', 'Customer_Satisfaction', 'Inventory_Level']
    
    # Categorical encoding
    ml_data_encoded = pd.get_dummies(ml_data, columns=['Product_Category', 'Customer_Segment', 
                                                      'Shipping_Mode', 'Order_Priority'], 
                                    prefix=['Cat', 'Seg', 'Ship', 'Pri'])
    
    # Select features (exclude datetime columns)
    feature_columns = [col for col in ml_data_encoded.columns if any(prefix in col for prefix in 
                      ['Order_Month', 'Order_Quarter', 'Order_DayOfWeek', 'Order_WeekOfYear',
                       'Unit_Price', 'Customer_Satisfaction', 'Inventory_Level', 'Cat_', 'Seg_', 'Ship_', 'Pri_'])]
    
    # Remove any datetime columns that might have been included
    feature_columns = [col for col in feature_columns if ml_data_encoded[col].dtype not in ['datetime64[ns]', 'object']]
    
    X = ml_data_encoded[feature_columns].fillna(0)
    y = ml_data_encoded['Quantity_Ordered']
    
    print(f"Using {len(feature_columns)} features for ML model")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    
    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    y_pred_lr = lr_model.predict(X_test)
    
    # Model evaluation
    print("\n📊 Model Performance:")
    print("Random Forest:")
    print(f"  MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
    print(f"  R²: {r2_score(y_test, y_pred_rf):.3f}")
    
    print("Linear Regression:")
    print(f"  MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
    print(f"  R²: {r2_score(y_test, y_pred_lr):.3f}")
    
    # Feature importance
    plt.figure(figsize=(15, 10))
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.subplot(2, 2, 1)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['Importance'], color='lightblue')
    plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=10)
    plt.title('Top 10 Feature Importance (Random Forest)', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Prediction vs Actual
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, y_pred_rf, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Demand', fontsize=12)
    plt.ylabel('Predicted Demand', fontsize=12)
    plt.title('Random Forest: Predicted vs Actual', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.scatter(y_test, y_pred_lr, alpha=0.6, color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Demand', fontsize=12)
    plt.ylabel('Predicted Demand', fontsize=12)
    plt.title('Linear Regression: Predicted vs Actual', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(2, 2, 4)
    residuals = y_test - y_pred_rf
    plt.scatter(y_pred_rf, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Demand', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residual Plot (Random Forest)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/ml_forecasting_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Machine Learning forecasting completed")
    return rf_model, feature_columns

def abc_analysis(df):
    """Perform ABC Analysis only"""
    print("\n" + "="*60)
    print("🏷️ ABC ANALYSIS")
    print("="*60)

    # ABC Analysis (Product Classification)
    product_analysis = df.groupby('Product_Name').agg({
        'Total_Sales': 'sum',
        'Quantity_Ordered': 'sum',
        'Profit': 'sum'
    }).reset_index()

    # Calculate cumulative percentages
    product_analysis = product_analysis.sort_values('Total_Sales', ascending=False)
    product_analysis['Cumulative_Sales'] = product_analysis['Total_Sales'].cumsum()
    product_analysis['Sales_Percentage'] = (product_analysis['Cumulative_Sales'] / 
                                           product_analysis['Total_Sales'].sum()) * 100

    # ABC Classification
    def classify_abc(percentage):
        if percentage <= 80:
            return 'A'
        elif percentage <= 95:
            return 'B'
        else:
            return 'C'

    product_analysis['ABC_Category'] = product_analysis['Sales_Percentage'].apply(classify_abc)

    # Visualization
    plt.figure(figsize=(18, 8))

    # ABC Analysis pie chart
    plt.subplot(1, 3, 1)
    abc_counts = product_analysis['ABC_Category'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.pie(abc_counts.values, labels=abc_counts.index, autopct='%1.1f%%', 
            textprops={'fontsize': 12}, colors=colors)
    plt.title('ABC Analysis - Product Distribution', fontsize=16)

    # Pareto Chart
    plt.subplot(1, 3, 2)
    top_n = 20
    plt.bar(range(top_n), product_analysis.head(top_n)['Total_Sales'], 
            color='skyblue', alpha=0.7)
    plt.plot(range(top_n), product_analysis.head(top_n)['Sales_Percentage'], 
             color='red', marker='o', linewidth=2)
    plt.title('Pareto Analysis - Top 20 Products', fontsize=16)
    plt.xlabel('Product Rank', fontsize=12)
    plt.ylabel('Sales / Cumulative %', fontsize=12)
    plt.xticks(range(0, top_n, 5), rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)

    # Category breakdown
    plt.subplot(1, 3, 3)
    category_breakdown = product_analysis.groupby('ABC_Category').agg({
        'Total_Sales': 'sum',
        'Product_Name': 'count'
    }).rename(columns={'Product_Name': 'Product_Count'})
    
    x_pos = range(len(category_breakdown))
    plt.bar(x_pos, category_breakdown['Product_Count'], color=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('Product Count by ABC Category', fontsize=16)
    plt.xlabel('ABC Category', fontsize=12)
    plt.ylabel('Number of Products', fontsize=12)
    plt.xticks(x_pos, category_breakdown.index)
    
    # Add value labels on bars
    for i, v in enumerate(category_breakdown['Product_Count']):
        plt.text(i, v + 0.01 * max(category_breakdown['Product_Count']), str(v), 
                ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('visualizations/abc_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print key insights
    print("\n📊 ABC ANALYSIS INSIGHTS:")
    print("-" * 40)
    for category in ['A', 'B', 'C']:
        count = len(product_analysis[product_analysis['ABC_Category'] == category])
        percentage = (count / len(product_analysis)) * 100
        sales_contribution = product_analysis[product_analysis['ABC_Category'] == category]['Total_Sales'].sum()
        total_sales = product_analysis['Total_Sales'].sum()
        sales_percentage = (sales_contribution / total_sales) * 100
        print(f"  Category {category}: {count} products ({percentage:.1f}%) - {sales_percentage:.1f}% of total sales")

    print("✓ ABC Analysis completed")
    return product_analysis

def turnover_analysis(df):
    """Perform Inventory Turnover Analysis only"""
    print("\n" + "="*60)
    print("📊 INVENTORY TURNOVER ANALYSIS")
    print("="*60)

    # Inventory Turnover Analysis
    inventory_metrics = df.groupby('Product_Category').agg({
        'Total_Sales': 'sum',
        'Total_Cost': 'sum',
        'Inventory_Level': 'mean',
        'Quantity_Ordered': 'sum',
        'Reorder_Point': 'mean',
        'On_Time_Delivery': 'mean'
    }).reset_index()

    # Calculate inventory turnover ratio
    inventory_metrics['Inventory_Turnover'] = inventory_metrics['Quantity_Ordered'] / inventory_metrics['Inventory_Level']
    inventory_metrics['Days_Supply'] = 365 / inventory_metrics['Inventory_Turnover']

    # Visualization
    plt.figure(figsize=(18, 6))

    # Inventory Turnover by Category
    plt.subplot(1, 3, 1)
    plt.barh(inventory_metrics['Product_Category'], inventory_metrics['Inventory_Turnover'], 
             color='lightgreen', alpha=0.8)
    plt.title('Inventory Turnover by Category', fontsize=16)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('Turnover Ratio', fontsize=12)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Days Supply
    plt.subplot(1, 3, 2)
    plt.barh(inventory_metrics['Product_Category'], inventory_metrics['Days_Supply'], 
             color='lightcoral', alpha=0.8)
    plt.title('Days Supply by Category', fontsize=16)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('Days Supply', fontsize=12)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Turnover vs Sales correlation
    plt.subplot(1, 3, 3)
    plt.scatter(inventory_metrics['Inventory_Turnover'], inventory_metrics['Total_Sales'], 
               color='purple', alpha=0.7, s=100)
    plt.xlabel('Inventory Turnover', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    plt.title('Turnover vs Sales Performance', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Add category labels
    for i, txt in enumerate(inventory_metrics['Product_Category']):
        plt.annotate(txt, (inventory_metrics['Inventory_Turnover'].iloc[i], 
                          inventory_metrics['Total_Sales'].iloc[i]), 
                    fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig('visualizations/turnover_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print key insights
    print("\n📊 TURNOVER ANALYSIS INSIGHTS:")
    print("-" * 40)
    best_turnover = inventory_metrics.loc[inventory_metrics['Inventory_Turnover'].idxmax()]
    worst_turnover = inventory_metrics.loc[inventory_metrics['Inventory_Turnover'].idxmin()]
    print(f"  Best turnover: {best_turnover['Product_Category']} ({best_turnover['Inventory_Turnover']:.2f})")
    print(f"  Worst turnover: {worst_turnover['Product_Category']} ({worst_turnover['Inventory_Turnover']:.2f})")
    print(f"  Average turnover: {inventory_metrics['Inventory_Turnover'].mean():.2f}")

    print("✓ Inventory Turnover Analysis completed")
    return inventory_metrics

def safety_stock_analysis(df):
    """Perform Safety Stock Calculation only"""
    print("\n" + "="*60)
    print("🛡️ SAFETY STOCK ANALYSIS")
    print("="*60)

    # Safety Stock Calculation
    def calculate_safety_stock(df_category):
        """Calculate safety stock based on demand variability"""
        daily_demand = df_category.groupby('Order_Date')['Quantity_Ordered'].sum()
        if len(daily_demand) == 0:
            return 0
        demand_std = daily_demand.std()
        service_level = 0.95  # 95% service level
        z_score = 1.65  # Z-score for 95% service level
        lead_time = 7  # Assumed lead time in days

        safety_stock = z_score * demand_std * np.sqrt(lead_time)
        return safety_stock if not np.isnan(safety_stock) else 0

    # Calculate safety stock for each category
    safety_stocks = []
    for category in df['Product_Category'].unique():
        category_data = df[df['Product_Category'] == category]
        safety_stock = calculate_safety_stock(category_data)
        
        # Calculate additional metrics
        avg_demand = category_data.groupby('Order_Date')['Quantity_Ordered'].sum().mean()
        demand_variability = category_data.groupby('Order_Date')['Quantity_Ordered'].sum().std()
        
        safety_stocks.append({
            'Product_Category': category,
            'Safety_Stock': safety_stock,
            'Average_Daily_Demand': avg_demand if not np.isnan(avg_demand) else 0,
            'Demand_Variability': demand_variability if not np.isnan(demand_variability) else 0
        })

    safety_stock_df = pd.DataFrame(safety_stocks)

    # Visualization
    plt.figure(figsize=(18, 6))

    # Safety Stock Requirements
    plt.subplot(1, 3, 1)
    plt.barh(safety_stock_df['Product_Category'], safety_stock_df['Safety_Stock'], 
             color='lightblue', alpha=0.8)
    plt.title('Safety Stock Requirements', fontsize=16)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('Safety Stock', fontsize=12)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Average Daily Demand
    plt.subplot(1, 3, 2)
    plt.barh(safety_stock_df['Product_Category'], safety_stock_df['Average_Daily_Demand'], 
             color='lightyellow', alpha=0.8)
    plt.title('Average Daily Demand', fontsize=16)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('Daily Demand', fontsize=12)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Demand Variability
    plt.subplot(1, 3, 3)
    plt.barh(safety_stock_df['Product_Category'], safety_stock_df['Demand_Variability'], 
             color='lightpink', alpha=0.8)
    plt.title('Demand Variability (Std Dev)', fontsize=16)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('Standard Deviation', fontsize=12)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/safety_stock_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print key insights
    print("\n📊 SAFETY STOCK INSIGHTS:")
    print("-" * 40)
    for _, row in safety_stock_df.iterrows():
        print(f"  {row['Product_Category']}: Safety Stock = {row['Safety_Stock']:.0f} units")
    
    highest_safety_stock = safety_stock_df.loc[safety_stock_df['Safety_Stock'].idxmax()]
    print(f"\n  Highest safety stock needed: {highest_safety_stock['Product_Category']} ({highest_safety_stock['Safety_Stock']:.0f} units)")

    print("✓ Safety Stock Analysis completed")
    return safety_stock_df

def inventory_optimization_analysis(df):
    """Perform inventory optimization analysis"""
    print("\n" + "="*60)
    print("📦 INVENTORY OPTIMIZATION ANALYSIS")
    print("="*60)

    # 1. ABC Analysis (Product Classification)
    product_analysis = df.groupby('Product_Name').agg({
        'Total_Sales': 'sum',
        'Quantity_Ordered': 'sum',
        'Profit': 'sum'
    }).reset_index()

    # Calculate cumulative percentages
    product_analysis = product_analysis.sort_values('Total_Sales', ascending=False)
    product_analysis['Cumulative_Sales'] = product_analysis['Total_Sales'].cumsum()
    product_analysis['Sales_Percentage'] = (product_analysis['Cumulative_Sales'] / 
                                           product_analysis['Total_Sales'].sum()) * 100

    # ABC Classification
    def classify_abc(percentage):
        if percentage <= 80:
            return 'A'
        elif percentage <= 95:
            return 'B'
        else:
            return 'C'

    product_analysis['ABC_Category'] = product_analysis['Sales_Percentage'].apply(classify_abc)

    # 2. Inventory Turnover Analysis
    inventory_metrics = df.groupby('Product_Category').agg({
        'Total_Sales': 'sum',
        'Total_Cost': 'sum',
        'Inventory_Level': 'mean',
        'Quantity_Ordered': 'sum',
        'Reorder_Point': 'mean',
        'On_Time_Delivery': 'mean'
    }).reset_index()

    # Calculate inventory turnover ratio
    inventory_metrics['Inventory_Turnover'] = inventory_metrics['Quantity_Ordered'] / inventory_metrics['Inventory_Level']
    inventory_metrics['Days_Supply'] = 365 / inventory_metrics['Inventory_Turnover']

    # 3. Safety Stock Calculation
    def calculate_safety_stock(df_category):
        """Calculate safety stock based on demand variability"""
        daily_demand = df_category.groupby('Order_Date')['Quantity_Ordered'].sum()
        demand_std = daily_demand.std()
        service_level = 0.95  # 95% service level
        z_score = 1.65  # Z-score for 95% service level
        lead_time = 7  # Assumed lead time in days

        safety_stock = z_score * demand_std * np.sqrt(lead_time)
        return safety_stock

    # Calculate safety stock for each category
    safety_stocks = []
    for category in df['Product_Category'].unique():
        category_data = df[df['Product_Category'] == category]
        safety_stock = calculate_safety_stock(category_data)
        safety_stocks.append({
            'Product_Category': category,
            'Safety_Stock': safety_stock
        })

    safety_stock_df = pd.DataFrame(safety_stocks)
    inventory_metrics = inventory_metrics.merge(safety_stock_df, on='Product_Category', how='left')

    # Optimal inventory level
    inventory_metrics['Optimal_Inventory'] = inventory_metrics['Reorder_Point'] + inventory_metrics['Safety_Stock']
    inventory_metrics['Current_vs_Optimal'] = inventory_metrics['Inventory_Level'] - inventory_metrics['Optimal_Inventory']

    # 4. Visualization
    plt.figure(figsize=(30, 20))  # Increase figure size for better readability

    # ABC Analysis
    plt.subplot(3, 3, 1)
    abc_counts = product_analysis['ABC_Category'].value_counts()
    plt.pie(abc_counts.values, labels=abc_counts.index, autopct='%1.1f%%', textprops={'fontsize': 12})
    plt.title('ABC Analysis - Product Distribution', fontsize=16)

    # Pareto Chart
    plt.subplot(3, 3, 2)
    top_n = 20
    plt.bar(range(top_n), product_analysis.head(top_n)['Total_Sales'], color='skyblue')
    plt.plot(range(top_n), product_analysis.head(top_n)['Sales_Percentage'], 
             color='red', marker='o')
    plt.title('Pareto Analysis - Top 20 Products', fontsize=16)
    plt.xlabel('Product Rank', fontsize=12)
    plt.ylabel('Sales / Cumulative %', fontsize=12)
    plt.xticks(range(top_n), product_analysis.head(top_n)['Product_Name'], rotation=90, fontsize=10)

    # Inventory Turnover by Category
    plt.subplot(3, 3, 3)
    plt.barh(inventory_metrics['Product_Category'], inventory_metrics['Inventory_Turnover'], color='lightgreen')
    plt.title('Inventory Turnover by Category', fontsize=16)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('Turnover Ratio', fontsize=12)
    plt.yticks(fontsize=10)

    # Days Supply
    plt.subplot(3, 3, 4)
    plt.barh(inventory_metrics['Product_Category'], inventory_metrics['Days_Supply'], color='lightcoral')
    plt.title('Days Supply by Category', fontsize=16)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('Days Supply', fontsize=12)
    plt.yticks(fontsize=10)

    # Current vs Optimal Inventory
    plt.subplot(3, 3, 5)
    plt.barh(inventory_metrics['Product_Category'], inventory_metrics['Current_vs_Optimal'], color='gold')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Current vs Optimal Inventory Levels', fontsize=16)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('Inventory Difference', fontsize=12)
    plt.yticks(fontsize=10)

    # Safety Stock Requirements
    plt.subplot(3, 3, 6)
    plt.barh(inventory_metrics['Product_Category'], inventory_metrics['Safety_Stock'], color='lightblue')
    plt.title('Safety Stock Requirements', fontsize=16)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('Safety Stock', fontsize=12)
    plt.yticks(fontsize=10)

    # Inventory vs Sales Correlation
    plt.subplot(3, 3, 7)
    plt.scatter(inventory_metrics['Inventory_Level'], inventory_metrics['Total_Sales'], color='purple', alpha=0.7)
    plt.xlabel('Average Inventory Level', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    plt.title('Inventory Level vs Sales Performance', fontsize=16)

    # On-Time Delivery by Category
    plt.subplot(3, 3, 8)
    plt.barh(inventory_metrics['Product_Category'], inventory_metrics['On_Time_Delivery'] * 100, color='orange')
    plt.title('On-Time Delivery Rate by Category', fontsize=16)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('On-Time Delivery %', fontsize=12)
    plt.yticks(fontsize=10)

    # Cost Analysis
    plt.subplot(3, 3, 9)
    cost_efficiency = inventory_metrics['Total_Sales'] / inventory_metrics['Total_Cost']
    plt.barh(inventory_metrics['Product_Category'], cost_efficiency, color='teal')
    plt.title('Sales-to-Cost Ratio by Category', fontsize=16)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('Sales/Cost Ratio', fontsize=12)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig('visualizations/inventory_optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print key insights
    print("\n📊 KEY INVENTORY INSIGHTS:")
    print("-" * 40)

    print("\n🏷️ ABC Analysis Results:")
    for category in ['A', 'B', 'C']:
        count = len(product_analysis[product_analysis['ABC_Category'] == category])
        percentage = (count / len(product_analysis)) * 100
        print(f"  Category {category}: {count} products ({percentage:.1f}%)")

    print("\n📦 Inventory Turnover Analysis:")
    best_turnover = inventory_metrics.loc[inventory_metrics['Inventory_Turnover'].idxmax()]
    worst_turnover = inventory_metrics.loc[inventory_metrics['Inventory_Turnover'].idxmin()]
    print(f"  Best turnover: {best_turnover['Product_Category']} ({best_turnover['Inventory_Turnover']:.2f})")
    print(f"  Worst turnover: {worst_turnover['Product_Category']} ({worst_turnover['Inventory_Turnover']:.2f})")

    print("\n⚡ Optimization Recommendations:")
    for _, row in inventory_metrics.iterrows():
        if row['Current_vs_Optimal'] > 50:
            print(f"  📉 Reduce inventory for {row['Product_Category']} by {row['Current_vs_Optimal']:.0f} units")
        elif row['Current_vs_Optimal'] < -50:
            print(f"  📈 Increase inventory for {row['Product_Category']} by {abs(row['Current_vs_Optimal']):.0f} units")

    return inventory_metrics, product_analysis

def supply_chain_kpi_dashboard(df):
    """Create a comprehensive KPI dashboard"""
    print("\n" + "="*60)
    print("📊 SUPPLY CHAIN KPI DASHBOARD")
    print("="*60)
    
    # Calculate KPIs
    kpis = {
        'Total Orders': len(df),
        'Total Revenue': df['Total_Sales'].sum(),
        'Total Profit': df['Profit'].sum(),
        'Average Order Value': df['Total_Sales'].mean(),
        'Profit Margin %': (df['Profit'].sum() / df['Total_Sales'].sum()) * 100,
        'On-Time Delivery %': df['On_Time_Delivery'].mean() * 100,
        'Average Customer Satisfaction': df['Customer_Satisfaction'].mean(),
        'Average Delivery Days': df['Delivery_Days'].mean(),
        'Total Products': df['Product_Name'].nunique(),
        'Active Suppliers': df['Supplier'].nunique(),
        'Countries Served': df['Country'].nunique()
    }
    
    # Monthly trends
    df_monthly = df.groupby(df['Order_Date'].dt.to_period('M')).agg({
        'Total_Sales': 'sum',
        'Profit': 'sum',
        'Quantity_Ordered': 'sum',
        'On_Time_Delivery': 'mean',
        'Customer_Satisfaction': 'mean'
    }).reset_index()
    
    df_monthly['Order_Date'] = df_monthly['Order_Date'].dt.to_timestamp()
    
    # Create dashboard
    plt.figure(figsize=(24, 18))
    
    # KPI Summary
    plt.subplot(3, 4, 1)
    kpi_names = list(kpis.keys())[:6]
    kpi_values = [kpis[name] for name in kpi_names]
    plt.barh(range(len(kpi_names)), kpi_values)
    plt.yticks(range(len(kpi_names)), kpi_names)
    plt.title('Key Performance Indicators')
    
    # Monthly Revenue Trend
    plt.subplot(3, 4, 2)
    plt.plot(df_monthly['Order_Date'], df_monthly['Total_Sales'], marker='o')
    plt.title('Monthly Revenue Trend')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.xticks(rotation=45)
    
    # Monthly Profit Trend
    plt.subplot(3, 4, 3)
    plt.plot(df_monthly['Order_Date'], df_monthly['Profit'], marker='s', color='green')
    plt.title('Monthly Profit Trend')
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.xticks(rotation=45)
    
    # On-Time Delivery Trend
    plt.subplot(3, 4, 4)
    plt.plot(df_monthly['Order_Date'], df_monthly['On_Time_Delivery'] * 100, marker='^', color='orange')
    plt.title('On-Time Delivery Trend')
    plt.xlabel('Date')
    plt.ylabel('On-Time Delivery %')
    plt.xticks(rotation=45)
    
    # Sales by Category
    plt.subplot(3, 4, 5)
    category_sales = df.groupby('Product_Category')['Total_Sales'].sum().sort_values(ascending=True)
    # Limit to top 8 categories to avoid overcrowding
    top_categories = category_sales.tail(8)
    colors = plt.cm.Set3(range(len(top_categories)))
    plt.barh(range(len(top_categories)), top_categories.values, color=colors)
    plt.yticks(range(len(top_categories)), [cat[:15] + '...' if len(cat) > 15 else cat for cat in top_categories.index], fontsize=10)
    plt.title('Top 8 Categories by Sales', fontsize=14)
    plt.xlabel('Total Sales', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Geographic Distribution
    plt.subplot(3, 4, 6)
    country_sales = df.groupby('Country')['Total_Sales'].sum().sort_values(ascending=False)
    # Limit to top 5 countries and group others
    top_countries = country_sales.head(5)
    if len(country_sales) > 5:
        others_sum = country_sales.iloc[5:].sum()
        top_countries['Others'] = others_sum
    
    colors = plt.cm.Pastel1(range(len(top_countries)))
    plt.pie(top_countries.values, labels=[label[:10] + '...' if len(label) > 10 else label for label in top_countries.index], 
            autopct='%1.1f%%', textprops={'fontsize': 10}, colors=colors)
    plt.title('Sales Distribution by Country', fontsize=14)
    
    # Customer Satisfaction by Segment
    plt.subplot(3, 4, 7)
    segment_satisfaction = df.groupby('Customer_Segment')['Customer_Satisfaction'].mean()
    colors = plt.cm.Set2(range(len(segment_satisfaction)))
    bars = plt.bar(segment_satisfaction.index, segment_satisfaction.values, color=colors, alpha=0.8)
    plt.title('Customer Satisfaction by Segment', fontsize=14)
    plt.xlabel('Customer Segment', fontsize=12)
    plt.ylabel('Average Satisfaction', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, segment_satisfaction.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Shipping Mode Performance
    plt.subplot(3, 4, 8)
    shipping_performance = df.groupby('Shipping_Mode').agg({
        'On_Time_Delivery': 'mean',
        'Customer_Satisfaction': 'mean'
    })
    
    x = range(len(shipping_performance))
    width = 0.35
    colors1 = plt.cm.Blues(0.7)
    colors2 = plt.cm.Oranges(0.7)
    plt.bar([i - width/2 for i in x], shipping_performance['On_Time_Delivery'], 
           width, label='On-Time %', color=colors1, alpha=0.8)
    plt.bar([i + width/2 for i in x], shipping_performance['Customer_Satisfaction']/5, 
           width, label='Satisfaction (scaled)', color=colors2, alpha=0.8)
    plt.xlabel('Shipping Mode', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.title('Shipping Mode Performance', fontsize=14)
    plt.xticks(x, [mode[:8] + '...' if len(mode) > 8 else mode for mode in shipping_performance.index], 
              rotation=45, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Supplier Performance
    plt.subplot(3, 4, 9)
    supplier_performance = df.groupby('Supplier').agg({
        'Total_Sales': 'sum',
        'On_Time_Delivery': 'mean'
    }).sort_values('Total_Sales', ascending=False)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(supplier_performance)))
    scatter = plt.scatter(supplier_performance['Total_Sales'], supplier_performance['On_Time_Delivery'] * 100, 
                         c=colors, alpha=0.7, s=80)
    plt.xlabel('Total Sales', fontsize=12)
    plt.ylabel('On-Time Delivery %', fontsize=12)
    plt.title('Supplier Performance Matrix', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Inventory Status Distribution
    plt.subplot(3, 4, 10)
    stock_status = df['Stock_Status'].value_counts()
    colors = ['#ff9999', '#ffcc99', '#99ff99']
    plt.pie(stock_status.values, labels=stock_status.index, autopct='%1.1f%%', 
           textprops={'fontsize': 11}, colors=colors[:len(stock_status)])
    plt.title('Inventory Status Distribution', fontsize=14)
    
    # Order Priority Analysis  
    plt.subplot(3, 4, 11)
    priority_metrics = df.groupby('Order_Priority').agg({
        'Delivery_Days': 'mean',
        'Customer_Satisfaction': 'mean'
    })
    
    x = range(len(priority_metrics))
    width = 0.35
    colors1 = plt.cm.Reds(0.7)
    colors2 = plt.cm.Greens(0.7)
    plt.bar([i - width/2 for i in x], priority_metrics['Delivery_Days'], 
           width, alpha=0.8, label='Avg Delivery Days', color=colors1)
    plt.bar([i + width/2 for i in x], priority_metrics['Customer_Satisfaction'], 
           width, alpha=0.8, label='Avg Satisfaction', color=colors2)
    plt.xlabel('Order Priority', fontsize=12)
    plt.ylabel('Days / Satisfaction', fontsize=12)
    plt.title('Priority vs Performance', fontsize=14)
    plt.xticks(x, priority_metrics.index, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Profit Margin by Category
    plt.subplot(3, 4, 12)
    category_profit = df.groupby('Product_Category').agg({
        'Profit': 'sum',
        'Total_Sales': 'sum'
    })
    category_profit['Profit_Margin'] = (category_profit['Profit'] / category_profit['Total_Sales']) * 100
    
    # Limit to top 8 categories
    top_profit_categories = category_profit.nlargest(8, 'Profit_Margin')
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_profit_categories)))
    
    plt.barh(range(len(top_profit_categories)), top_profit_categories['Profit_Margin'], color=colors)
    plt.ylabel('Product Category', fontsize=12)
    plt.xlabel('Profit Margin %', fontsize=12)
    plt.title('Top 8 Categories - Profit Margin', fontsize=14)
    plt.yticks(range(len(top_profit_categories)), 
              [cat[:12] + '...' if len(cat) > 12 else cat for cat in top_profit_categories.index], fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/supply_chain_kpi_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print KPI summary
    print("\n📊 SUPPLY CHAIN KPIs:")
    print("-" * 30)
    for kpi, value in kpis.items():
        if isinstance(value, float):
            print(f"{kpi}: {value:,.2f}")
        else:
            print(f"{kpi}: {value:,}")
    
    return kpis

def performance_metrics_only(df):
    """Create simplified performance metrics display"""
    print("\n" + "="*60)
    print("📈 PERFORMANCE METRICS SUMMARY")
    print("="*60)
    
    # Calculate key metrics
    kpis = {
        'Total Orders': len(df),
        'Total Revenue': df['Total_Sales'].sum(),
        'Total Profit': df['Profit'].sum(),
        'Average Order Value': df['Total_Sales'].mean(),
        'Profit Margin %': (df['Profit'].sum() / df['Total_Sales'].sum()) * 100,
        'On-Time Delivery %': df['On_Time_Delivery'].mean() * 100,
        'Average Customer Satisfaction': df['Customer_Satisfaction'].mean(),
        'Average Delivery Days': df['Delivery_Days'].mean(),
        'Total Products': df['Product_Name'].nunique(),
        'Active Suppliers': df['Supplier'].nunique(),
        'Countries Served': df['Country'].nunique()
    }
    
    # Create a simple visualization
    plt.figure(figsize=(16, 10))
    
    # Key metrics bar chart
    plt.subplot(2, 2, 1)
    metric_names = ['Total Orders', 'Average Order Value', 'Profit Margin %', 'On-Time Delivery %']
    metric_values = [kpis[name] for name in metric_names]
    colors = ['skyblue', 'lightgreen', 'gold', 'orange']
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8)
    plt.title('Key Performance Indicators', fontsize=16)
    plt.ylabel('Values', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Revenue vs Profit comparison
    plt.subplot(2, 2, 2)
    categories = ['Revenue', 'Profit']
    values = [kpis['Total Revenue'], kpis['Total Profit']]
    plt.bar(categories, values, color=['lightblue', 'lightcoral'], alpha=0.8)
    plt.title('Revenue vs Profit', fontsize=16)
    plt.ylabel('Amount', fontsize=12)
    
    # Customer satisfaction by segment
    plt.subplot(2, 2, 3)
    segment_satisfaction = df.groupby('Customer_Segment')['Customer_Satisfaction'].mean()
    plt.bar(segment_satisfaction.index, segment_satisfaction.values, 
           color='mediumpurple', alpha=0.8)
    plt.title('Customer Satisfaction by Segment', fontsize=16)
    plt.xlabel('Customer Segment', fontsize=12)
    plt.ylabel('Average Satisfaction', fontsize=12)
    plt.xticks(rotation=45)
    
    # Top performing categories
    plt.subplot(2, 2, 4)
    top_categories = df.groupby('Product_Category')['Total_Sales'].sum().nlargest(5)
    plt.barh(range(len(top_categories)), top_categories.values, color='lightgreen', alpha=0.8)
    plt.yticks(range(len(top_categories)), top_categories.index, fontsize=10)
    plt.title('Top 5 Categories by Sales', fontsize=16)
    plt.xlabel('Total Sales', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualizations/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n📊 PERFORMANCE SUMMARY:")
    print("-" * 40)
    for kpi, value in kpis.items():
        if isinstance(value, float):
            print(f"  {kpi}: {value:,.2f}")
        else:
            print(f"  {kpi}: {value:,}")
    
    return kpis

def main():
    """Main function for advanced supply chain analysis"""
    print("🚀 Welcome to the Advanced Supply Chain Optimization Tool!")
    print("=" * 70)
    print("This tool helps you analyze and optimize your supply chain with ease.")
    print("Let's get started!\n")

    # Create visualizations directory
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    # Load and prepare data
    print("📊 Loading and preparing your dataset...")
    df = load_and_prepare_data()
    print("✓ Dataset is ready for analysis!\n")

    while True:
        print("\nWhat would you like to do today?")
        print("1️⃣  Perform Demand Forecasting Analysis")
        print("2️⃣  Conduct Inventory Optimization Analysis")
        print("3️⃣  Generate a Supply Chain KPI Dashboard")
        print("4️⃣  Exit the Tool")

        choice = input("👉 Please enter your choice (1-4): ").strip()

        if choice == '1':
            while True:
                print("\n🔮 Demand Forecasting Options:")
                print("a. Time Series Analysis")
                print("b. Machine Learning Forecasting")
                print("c. Back to Main Menu")

                sub_choice = input("👉 Please enter your choice (a-c): ").strip()

                if sub_choice == 'a':
                    print("\n📈 Performing Time Series Analysis...")
                    time_series_analysis(df)  # Call the function for time series analysis
                    print("✓ Time Series Analysis completed!\n")
                elif sub_choice == 'b':
                    print("\n🤖 Performing Machine Learning Forecasting...")
                    ml_forecasting_analysis(df)  # Call the function for ML forecasting
                    print("✓ Machine Learning Forecasting completed!\n")
                elif sub_choice == 'c':
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")

        elif choice == '2':
            while True:
                print("\n📦 Inventory Optimization Options:")
                print("a. ABC Analysis")
                print("b. Inventory Turnover Analysis")
                print("c. Safety Stock Calculation")
                print("d. Back to Main Menu")

                sub_choice = input("👉 Please enter your choice (a-d): ").strip()

                if sub_choice == 'a':
                    print("\n🏷️ Performing ABC Analysis...")
                    abc_analysis(df)  # Call the function for ABC analysis
                    print("✓ ABC Analysis completed!\n")
                elif sub_choice == 'b':
                    print("\n📊 Performing Inventory Turnover Analysis...")
                    turnover_analysis(df)  # Call the function for turnover analysis
                    print("✓ Inventory Turnover Analysis completed!\n")
                elif sub_choice == 'c':
                    print("\n🛡️ Calculating Safety Stock...")
                    safety_stock_analysis(df)  # Call the function for safety stock calculation
                    print("✓ Safety Stock Calculation completed!\n")
                elif sub_choice == 'd':
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")

        elif choice == '3':
            while True:
                print("\n📊 KPI Dashboard Options:")
                print("a. Complete KPI Dashboard")
                print("b. Performance Metrics Only")
                print("c. Back to Main Menu")

                sub_choice = input("👉 Please enter your choice (a-c): ").strip()

                if sub_choice == 'a':
                    print("\n📊 Generating Complete Supply Chain KPI Dashboard...")
                    supply_chain_kpi_dashboard(df)  # Call the function for KPI dashboard
                    print("🎉 Complete KPI Dashboard generated successfully!\n")
                elif sub_choice == 'b':
                    print("\n📈 Generating Performance Metrics...")
                    performance_metrics_only(df)  # Call simplified metrics function
                    print("🎉 Performance Metrics generated successfully!\n")
                elif sub_choice == 'c':
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")
        elif choice == '4':
            while True:
                print("\n🎯 Additional Options:")
                print("a. Generate Strategic Recommendations")
                print("b. Export Analysis Results")
                print("c. Back to Main Menu")

                sub_choice = input("👉 Please enter your choice (a-c): ").strip()

                if sub_choice == 'a':
                    print("\n🎯 Generating Strategic Recommendations...")
                    generate_recommendations(df)
                    print("✓ Strategic Recommendations generated!\n")
                elif sub_choice == 'b':
                    print("\n💾 Exporting Analysis Results...")
                    export_results(df)
                    print("✓ Results exported successfully!\n")
                elif sub_choice == 'c':
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")
        elif choice == '5':
            print("\n👋 Thank you for using the Advanced Supply Chain Optimization Tool.")
            print("Have a great day! Goodbye! 👋")
            break
        else:
            print("\n❌ Oops! That doesn't seem like a valid option. Please try again.")

def generate_recommendations(df):
    """Generate strategic recommendations based on analysis"""
    print("\n" + "="*70)
    print("🎯 STRATEGIC RECOMMENDATIONS")
    print("="*70)
    
    # Analyze current performance
    avg_delivery_days = df['Delivery_Days'].mean()
    on_time_rate = df['On_Time_Delivery'].mean()
    profit_margin = (df['Profit'].sum() / df['Total_Sales'].sum()) * 100
    customer_satisfaction = df['Customer_Satisfaction'].mean()
    
    print("\n📈 DEMAND FORECASTING RECOMMENDATIONS:")
    print("• Implement machine learning models for demand prediction")
    print("• Monitor seasonal patterns and adjust inventory accordingly")
    print("• Focus on high-impact features identified in the analysis")
    
    print("\n📦 INVENTORY OPTIMIZATION RECOMMENDATIONS:")
    print("• Prioritize Category A products for inventory investment")
    print("• Implement dynamic safety stock calculations")
    print("• Optimize reorder points based on demand variability")
    
    print("\n🚛 SUPPLY CHAIN EFFICIENCY RECOMMENDATIONS:")
    if on_time_rate < 0.85:
        print("• URGENT: Improve on-time delivery rates through supplier partnerships")
    else:
        print("• Maintain good on-time delivery performance")
    
    if avg_delivery_days > 10:
        print("• FOCUS: Reduce average delivery time - currently too high")
    else:
        print("• Good delivery time performance - maintain current levels")
    
    print("• Optimize shipping modes based on customer requirements")
    print("• Focus on high-performing suppliers and regions")
    
    print("\n💰 COST OPTIMIZATION RECOMMENDATIONS:")
    if profit_margin < 10:
        print("• CRITICAL: Improve profit margins - currently below 10%")
    else:
        print("• Good profit margin performance")
    
    print("• Reduce inventory holding costs for slow-moving items")
    print("• Implement just-in-time delivery for high-turnover products")
    print("• Negotiate better terms with suppliers based on volume")
    
    print("\n👥 CUSTOMER EXPERIENCE RECOMMENDATIONS:")
    if customer_satisfaction < 3.5:
        print("• URGENT: Address customer satisfaction issues")
    else:
        print("• Maintain good customer satisfaction levels")
    
    print("• Implement customer feedback loops")
    print("• Focus on high-value customer segments")

def export_results(df):
    """Export analysis results to files"""
    print("\n💾 Exporting analysis results...")
    
    try:
        # Create exports directory
        if not os.path.exists('exports'):
            os.makedirs('exports')
        
        # Export basic statistics
        basic_stats = df.describe()
        basic_stats.to_csv('exports/basic_statistics.csv')
        
        # Export category analysis
        category_analysis = df.groupby('Product_Category').agg({
            'Total_Sales': 'sum',
            'Profit': 'sum',
            'Quantity_Ordered': 'sum',
            'On_Time_Delivery': 'mean',
            'Customer_Satisfaction': 'mean'
        }).round(2)
        category_analysis.to_csv('exports/category_analysis.csv')
        
        # Export supplier performance
        supplier_performance = df.groupby('Supplier').agg({
            'Total_Sales': 'sum',
            'On_Time_Delivery': 'mean',
            'Customer_Satisfaction': 'mean'
        }).round(2)
        supplier_performance.to_csv('exports/supplier_performance.csv')
        
        # Export monthly trends
        monthly_trends = df.groupby(df['Order_Date'].dt.to_period('M')).agg({
            'Total_Sales': 'sum',
            'Profit': 'sum',
            'Quantity_Ordered': 'sum'
        }).round(2)
        monthly_trends.to_csv('exports/monthly_trends.csv')
        
        print("✓ Basic statistics exported to 'exports/basic_statistics.csv'")
        print("✓ Category analysis exported to 'exports/category_analysis.csv'")
        print("✓ Supplier performance exported to 'exports/supplier_performance.csv'")
        print("✓ Monthly trends exported to 'exports/monthly_trends.csv'")
        
    except Exception as e:
        print(f"❌ Error exporting results: {str(e)}")

def main():
    """Main function for advanced supply chain analysis"""
    print("🚀 Welcome to the Advanced Supply Chain Optimization Tool!")
    print("=" * 70)
    print("This tool helps you analyze and optimize your supply chain with ease.")
    print("Let's get started!\n")

    # Create visualizations directory
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    # Load and prepare data
    print("📊 Loading and preparing your dataset...")
    df = load_and_prepare_data()
    print("✓ Dataset is ready for analysis!\n")

    while True:
        print("\nWhat would you like to do today?")
        print("1️⃣  Perform Demand Forecasting Analysis")
        print("2️⃣  Conduct Inventory Optimization Analysis")
        print("3️⃣  Generate Supply Chain KPI Dashboard")
        print("4️⃣  Additional Analysis Options")
        print("5️⃣  Exit the Tool")

        choice = input("👉 Please enter your choice (1-5): ").strip()

        if choice == '1':
            while True:
                print("\n🔮 Demand Forecasting Options:")
                print("a. Time Series Analysis")
                print("b. Machine Learning Forecasting")
                print("c. Back to Main Menu")

                sub_choice = input("👉 Please enter your choice (a-c): ").strip()

                if sub_choice == 'a':
                    print("\n📈 Performing Time Series Analysis...")
                    time_series_analysis(df)  # Call the function for time series analysis
                    print("✓ Time Series Analysis completed!\n")
                elif sub_choice == 'b':
                    print("\n🤖 Performing Machine Learning Forecasting...")
                    ml_forecasting_analysis(df)  # Call the function for ML forecasting
                    print("✓ Machine Learning Forecasting completed!\n")
                elif sub_choice == 'c':
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")

        elif choice == '2':
            while True:
                print("\n📦 Inventory Optimization Options:")
                print("a. ABC Analysis")
                print("b. Inventory Turnover Analysis")
                print("c. Safety Stock Calculation")
                print("d. Back to Main Menu")

                sub_choice = input("👉 Please enter your choice (a-d): ").strip()

                if sub_choice == 'a':
                    print("\n🏷️ Performing ABC Analysis...")
                    abc_analysis(df)  # Call the function for ABC analysis
                    print("✓ ABC Analysis completed!\n")
                elif sub_choice == 'b':
                    print("\n📊 Performing Inventory Turnover Analysis...")
                    turnover_analysis(df)  # Call the function for turnover analysis
                    print("✓ Inventory Turnover Analysis completed!\n")
                elif sub_choice == 'c':
                    print("\n🛡️ Calculating Safety Stock...")
                    safety_stock_analysis(df)  # Call the function for safety stock calculation
                    print("✓ Safety Stock Calculation completed!\n")
                elif sub_choice == 'd':
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")

        elif choice == '3':
            while True:
                print("\n📊 KPI Dashboard Options:")
                print("a. Complete KPI Dashboard")
                print("b. Performance Metrics Only")
                print("c. Back to Main Menu")

                sub_choice = input("👉 Please enter your choice (a-c): ").strip()

                if sub_choice == 'a':
                    print("\n📊 Generating Complete Supply Chain KPI Dashboard...")
                    supply_chain_kpi_dashboard(df)  # Call the function for KPI dashboard
                    print("🎉 Complete KPI Dashboard generated successfully!\n")
                elif sub_choice == 'b':
                    print("\n📈 Generating Performance Metrics...")
                    performance_metrics_only(df)  # Call simplified metrics function
                    print("🎉 Performance Metrics generated successfully!\n")
                elif sub_choice == 'c':
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")

        elif choice == '4':
            while True:
                print("\n🎯 Additional Options:")
                print("a. Generate Strategic Recommendations")
                print("b. Export Analysis Results")
                print("c. Back to Main Menu")

                sub_choice = input("👉 Please enter your choice (a-c): ").strip()

                if sub_choice == 'a':
                    print("\n🎯 Generating Strategic Recommendations...")
                    generate_recommendations(df)
                    print("✓ Strategic Recommendations generated!\n")
                elif sub_choice == 'b':
                    print("\n💾 Exporting Analysis Results...")
                    export_results(df)
                    print("✓ Results exported successfully!\n")
                elif sub_choice == 'c':
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")
        elif choice == '5':
            print("\n👋 Thank you for using the Advanced Supply Chain Optimization Tool.")
            print("Have a great day! Goodbye! 👋")
            break
        else:
            print("\n❌ Oops! That doesn't seem like a valid option. Please try again.")

if __name__ == "__main__":
    main()
