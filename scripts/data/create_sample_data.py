"""
Create sample data for the recommendation engine.

This script generates synthetic data for testing and development.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create output directories if they don't exist
os.makedirs('data/processed', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Generate user data
def generate_users(num_users=100):
    users = []
    for i in range(num_users):
        user_id = f"user_{i+1}"
        username = f"user_{i+1}"
        email = f"user_{i+1}@example.com"
        join_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
        
        users.append({
            'user_id': user_id,
            'username': username,
            'email': email,
            'join_date': join_date,
            'last_active': join_date + timedelta(days=np.random.randint(0, 30))
        })
    
    return pd.DataFrame(users)

# Generate product data
def generate_products(num_products=500):
    categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Kitchen', 'Sports']
    subcategories = {
        'Electronics': ['Phones', 'Laptops', 'Tablets', 'Cameras', 'Headphones'],
        'Books': ['Fiction', 'Non-Fiction', 'Science', 'History', 'Biography'],
        'Clothing': ['Men', 'Women', 'Kids', 'Accessories', 'Shoes'],
        'Home': ['Furniture', 'Decor', 'Bedding', 'Lighting', 'Storage'],
        'Kitchen': ['Appliances', 'Cookware', 'Utensils', 'Dinnerware', 'Gadgets'],
        'Sports': ['Fitness', 'Outdoor', 'Team Sports', 'Water Sports', 'Winter Sports']
    }
    
    products = []
    for i in range(num_products):
        product_id = f"product_{i+1}"
        category = np.random.choice(categories)
        subcategory = np.random.choice(subcategories[category])
        title = f"{subcategory} Item {i+1}"
        description = f"This is a {subcategory.lower()} item in the {category.lower()} category."
        price = round(np.random.uniform(10.0, 200.0), 2)
        
        products.append({
            'product_id': product_id,
            'title': title,
            'description': description,
            'price': price,
            'category': category,
            'subcategory': subcategory,
            'created_at': datetime.now() - timedelta(days=np.random.randint(1, 730))
        })
    
    return pd.DataFrame(products)

# Generate review data
def generate_reviews(users_df, products_df, num_reviews=5000):
    reviews = []
    
    user_ids = users_df['user_id'].tolist()
    product_ids = products_df['product_id'].tolist()
    
    # Create a set to avoid duplicate user-product pairs
    user_product_pairs = set()
    
    for i in range(num_reviews):
        user_id = np.random.choice(user_ids)
        product_id = np.random.choice(product_ids)
        
        # Skip if we already have this pair
        pair = (user_id, product_id)
        if pair in user_product_pairs:
            continue
        
        user_product_pairs.add(pair)
        
        rating = np.random.randint(1, 6)  # 1-5 stars
        verified_purchase = np.random.choice([True, False], p=[0.8, 0.2])
        review_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
        votes = np.random.randint(0, 50)
        
        # Generate a review text based on rating
        if rating >= 4:
            review_text = "I really liked this product. It exceeded my expectations."
            summary = "Great product!"
        elif rating == 3:
            review_text = "This product is okay. It meets the basic requirements."
            summary = "Decent product"
        else:
            review_text = "I was disappointed with this product. It did not meet my expectations."
            summary = "Disappointing"
        
        reviews.append({
            'user_id': user_id,
            'product_id': product_id,
            'rating': rating,
            'review_text': review_text,
            'summary': summary,
            'verified_purchase': verified_purchase,
            'review_date': review_date,
            'votes': votes
        })
    
    return pd.DataFrame(reviews)

# Generate events data
def generate_events(users_df, products_df, num_events=10000):
    events = []
    
    user_ids = users_df['user_id'].tolist()
    product_ids = products_df['product_id'].tolist()
    event_types = ['view', 'click', 'add_to_cart', 'purchase', 'rate']
    
    for i in range(num_events):
        user_id = np.random.choice(user_ids)
        product_id = np.random.choice(product_ids)
        event_type = np.random.choice(event_types)
        event_timestamp = datetime.now() - timedelta(days=np.random.randint(1, 100))
        session_id = f"session_{np.random.randint(1, 1000)}"
        
        event_data = {
            'device': np.random.choice(['mobile', 'desktop', 'tablet']),
            'browser': np.random.choice(['chrome', 'firefox', 'safari', 'edge']),
            'referrer': np.random.choice(['direct', 'search', 'social', 'email']),
            'duration': np.random.randint(5, 300)
        }
        
        events.append({
            'user_id': user_id,
            'product_id': product_id,
            'event_type': event_type,
            'event_timestamp': event_timestamp,
            'session_id': session_id,
            'event_data': event_data
        })
    
    return pd.DataFrame(events)

# Create DataFrames
print("Generating users...")
users_df = generate_users(100)

print("Generating products...")
products_df = generate_products(500)

print("Generating reviews...")
reviews_df = generate_reviews(users_df, products_df, 5000)

print("Generating events...")
events_df = generate_events(users_df, products_df, 10000)

# Create features dataframe for model training
print("Creating features dataframe...")
features_df = reviews_df[['user_id', 'product_id', 'rating']].copy()

# Save DataFrames
print("Saving data...")
users_df.to_parquet('data/processed/users.parquet', index=False)
products_df.to_parquet('data/processed/products.parquet', index=False)
reviews_df.to_parquet('data/processed/reviews.parquet', index=False)
events_df.to_parquet('data/processed/events.parquet', index=False)
features_df.to_parquet('data/processed/features.parquet', index=False)

print("Sample data generation complete!")
print(f"- {len(users_df)} users")
print(f"- {len(products_df)} products")
print(f"- {len(reviews_df)} reviews")
print(f"- {len(events_df)} events")
print(f"- {len(features_df)} features") 