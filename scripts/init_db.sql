-- Create database and user (if they don't exist yet)
-- Note: This is already handled by Docker Compose

-- Create schema
CREATE SCHEMA IF NOT EXISTS amazon_rec;

-- Set search path
SET search_path TO amazon_rec;

-- Create products table
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(1000),
    description TEXT,
    price NUMERIC(10, 2),
    category VARCHAR(255),
    subcategory VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    username VARCHAR(255),
    email VARCHAR(255),
    join_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create reviews table
CREATE TABLE IF NOT EXISTS reviews (
    review_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    product_id VARCHAR(255) REFERENCES products(product_id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review_text TEXT,
    summary VARCHAR(1000),
    verified_purchase BOOLEAN DEFAULT FALSE,
    review_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    votes INTEGER DEFAULT 0
);

-- Create events table
CREATE TABLE IF NOT EXISTS events (
    event_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    product_id VARCHAR(255) REFERENCES products(product_id),
    event_type VARCHAR(50) CHECK (event_type IN ('view', 'add_to_cart', 'purchase', 'recommend_click')),
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(255),
    metadata JSONB
);

-- Create recommendations table
CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    product_id VARCHAR(255) REFERENCES products(product_id),
    score NUMERIC(10, 6),
    model_version VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    served_at TIMESTAMP WITH TIME ZONE,
    clicked BOOLEAN DEFAULT FALSE,
    purchased BOOLEAN DEFAULT FALSE
);

-- Create model_versions table
CREATE TABLE IF NOT EXISTS model_versions (
    model_version_id SERIAL PRIMARY KEY,
    model_name VARCHAR(255),
    model_version VARCHAR(255),
    artifact_path VARCHAR(1000),
    training_time NUMERIC,
    model_parameters JSONB,
    metrics JSONB,
    is_production BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create experiments table
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    start_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) CHECK (status IN ('running', 'completed', 'stopped')),
    control_model_version_id INTEGER REFERENCES model_versions(model_version_id),
    treatment_model_version_id INTEGER REFERENCES model_versions(model_version_id),
    metrics JSONB,
    results JSONB
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews(user_id);
CREATE INDEX IF NOT EXISTS idx_reviews_product_id ON reviews(product_id);
CREATE INDEX IF NOT EXISTS idx_events_user_id ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_product_id ON events(product_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations(user_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_product_id ON recommendations(product_id);

-- Create a view for user-item ratings matrix
CREATE OR REPLACE VIEW user_item_ratings AS
SELECT user_id, product_id, rating
FROM reviews;

-- Create a view for item popularity
CREATE OR REPLACE VIEW item_popularity AS
SELECT product_id, COUNT(*) as view_count
FROM events
WHERE event_type = 'view'
GROUP BY product_id;

-- Create a view for user activity
CREATE OR REPLACE VIEW user_activity AS
SELECT user_id, COUNT(*) as event_count, 
       SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchase_count
FROM events
GROUP BY user_id;

-- Function to update the 'updated_at' field
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updating 'updated_at'
CREATE TRIGGER update_product_updated_at BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create a role for the recommendation service
CREATE ROLE recommendation_service WITH LOGIN PASSWORD 'recommendation_service_password';

-- Grant privileges
GRANT USAGE ON SCHEMA amazon_rec TO recommendation_service;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA amazon_rec TO recommendation_service;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA amazon_rec TO recommendation_service; 