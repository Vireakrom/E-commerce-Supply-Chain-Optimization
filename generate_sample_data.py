# Sample Data Generator for Supply Chain Optimization
# This creates sample data for testing if you don't have the Kaggle dataset yet

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_supply_chain_data(num_records=10000):
    """
    Generate sample supply chain data for testing
    """
    print("Generating sample supply chain data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Product categories and names
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys', 'Health', 'Beauty']
    products = [
        'Smartphone', 'Laptop', 'Headphones', 'Camera', 'Tablet',
        'T-Shirt', 'Jeans', 'Sneakers', 'Jacket', 'Dress',
        'Chair', 'Table', 'Lamp', 'Curtains', 'Pillow',
        'Soccer Ball', 'Tennis Racket', 'Running Shoes', 'Gym Equipment', 'Bicycle',
        'Novel', 'Textbook', 'Magazine', 'Comic Book', 'Dictionary',
        'Action Figure', 'Board Game', 'Puzzle', 'Doll', 'Building Blocks',
        'Vitamins', 'Medicine', 'First Aid Kit', 'Thermometer', 'Supplements',
        'Lipstick', 'Shampoo', 'Perfume', 'Face Cream', 'Nail Polish'
    ]
    
    # Suppliers and warehouses
    suppliers = ['Supplier_A', 'Supplier_B', 'Supplier_C', 'Supplier_D', 'Supplier_E']
    warehouses = ['Warehouse_North', 'Warehouse_South', 'Warehouse_East', 'Warehouse_West', 'Warehouse_Central']
    
    # Countries and cities
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan', 'Singapore']
    cities = ['New York', 'Toronto', 'London', 'Berlin', 'Paris', 'Sydney', 'Tokyo', 'Singapore']
    
    # Shipping modes
    shipping_modes = ['Standard', 'Express', 'Overnight', 'Economy']
    
    # Order priorities
    priorities = ['Low', 'Medium', 'High', 'Critical']
    
    # Customer segments
    customer_segments = ['Corporate', 'Consumer', 'Home Office', 'Small Business']
    
    # Generate the dataset
    data = []
    
    for i in range(num_records):
        # Random date
        order_date = random.choice(date_range)
        ship_date = order_date + timedelta(days=random.randint(1, 7))
        delivery_date = ship_date + timedelta(days=random.randint(1, 14))
        
        # Random product
        category = random.choice(categories)
        product = random.choice(products)
        
        # Random quantities and prices
        quantity = random.randint(1, 100)
        unit_price = round(random.uniform(10, 1000), 2)
        total_sales = round(quantity * unit_price, 2)
        
        # Random costs
        unit_cost = round(unit_price * random.uniform(0.3, 0.7), 2)
        total_cost = round(quantity * unit_cost, 2)
        profit = round(total_sales - total_cost, 2)
        
        # Random logistics
        supplier = random.choice(suppliers)
        warehouse = random.choice(warehouses)
        country = random.choice(countries)
        city = random.choice(cities)
        shipping_mode = random.choice(shipping_modes)
        priority = random.choice(priorities)
        customer_segment = random.choice(customer_segments)
        
        # Random performance metrics
        delivery_days = (delivery_date - order_date).days
        on_time_delivery = 1 if delivery_days <= 7 else 0
        customer_satisfaction = round(random.uniform(1, 5), 1)
        
        # Create record
        record = {
            'Order_ID': f'ORD_{i+1:06d}',
            'Order_Date': order_date.strftime('%Y-%m-%d'),
            'Ship_Date': ship_date.strftime('%Y-%m-%d'),
            'Delivery_Date': delivery_date.strftime('%Y-%m-%d'),
            'Product_Category': category,
            'Product_Name': product,
            'Supplier': supplier,
            'Warehouse': warehouse,
            'Country': country,
            'City': city,
            'Customer_Segment': customer_segment,
            'Shipping_Mode': shipping_mode,
            'Order_Priority': priority,
            'Quantity_Ordered': quantity,
            'Unit_Price': unit_price,
            'Unit_Cost': unit_cost,
            'Total_Sales': total_sales,
            'Total_Cost': total_cost,
            'Profit': profit,
            'Delivery_Days': delivery_days,
            'On_Time_Delivery': on_time_delivery,
            'Customer_Satisfaction': customer_satisfaction,
            'Inventory_Level': random.randint(0, 1000),
            'Reorder_Point': random.randint(50, 200),
            'Stock_Status': random.choice(['In Stock', 'Low Stock', 'Out of Stock'])
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    filename = 'sample_supply_chain_data.csv'
    df.to_csv(filename, index=False)
    
    print(f"✓ Sample dataset created: {filename}")
    print(f"✓ Records generated: {len(df):,}")
    print(f"✓ Columns: {len(df.columns)}")
    print(f"✓ Date range: {df['Order_Date'].min()} to {df['Order_Date'].max()}")
    
    return df

if __name__ == "__main__":
    # Generate sample data
    sample_df = generate_sample_supply_chain_data(10000)
    
    print("\nSample data preview:")
    print(sample_df.head())
    
    print("\nDataset summary:")
    print(sample_df.info())
