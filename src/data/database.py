"""
Database connection and ORM models for the Amazon recommendation system.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                       String, Text, create_engine, func, text)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import QueuePool

# Set up logging
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = os.environ.get('CONFIG_PATH', 'config/config.yaml')

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config from {CONFIG_PATH}: {e}")
        # Provide default config for database
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'amazon_rec_db',
                'user': 'postgres',
                'password': 'postgres',
                'schema': 'amazon_rec',
                'pool_size': 5,
                'max_overflow': 10,
                'pool_recycle': 3600,
            }
        }

config = load_config()
db_config = config['database']

# Create SQLAlchemy engine and session
def get_connection_string() -> str:
    """Get database connection string from config."""
    return (f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}")

# Create engine with connection pooling
engine = create_engine(
    get_connection_string(),
    poolclass=QueuePool,
    pool_size=db_config.get('pool_size', 5),
    max_overflow=db_config.get('max_overflow', 10),
    pool_recycle=db_config.get('pool_recycle', 3600),
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize base class for SQLAlchemy models
Base = declarative_base()

# Define ORM models
class Product(Base):
    """Product model for database."""
    __tablename__ = "products"
    __table_args__ = {"schema": db_config['schema']}

    product_id = Column(String(255), primary_key=True)
    title = Column(String(1000))
    description = Column(Text)
    price = Column(Float)
    category = Column(String(255))
    subcategory = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    reviews = relationship("Review", back_populates="product")
    events = relationship("Event", back_populates="product")
    recommendations = relationship("Recommendation", back_populates="product")

    def get_id_column(self):
        """Get ID column for CRUD operations."""
        return Product.product_id


class User(Base):
    """User model for database."""
    __tablename__ = "users"
    __table_args__ = {"schema": db_config['schema']}

    user_id = Column(String(255), primary_key=True)
    username = Column(String(255))
    email = Column(String(255))
    join_date = Column(DateTime, default=func.now())
    last_active = Column(DateTime, default=func.now())

    # Relationships
    reviews = relationship("Review", back_populates="user")
    events = relationship("Event", back_populates="user")
    recommendations = relationship("Recommendation", back_populates="user")

    def get_id_column(self):
        """Get ID column for CRUD operations."""
        return User.user_id


class Review(Base):
    """Review model for database."""
    __tablename__ = "reviews"
    __table_args__ = {"schema": db_config['schema']}

    review_id = Column(Integer, primary_key=True)
    user_id = Column(String(255), ForeignKey(f"{db_config['schema']}.users.user_id"))
    product_id = Column(String(255), ForeignKey(f"{db_config['schema']}.products.product_id"))
    rating = Column(Integer)
    review_text = Column(Text)
    summary = Column(String(1000))
    verified_purchase = Column(Boolean, default=False)
    review_date = Column(DateTime, default=func.now())
    votes = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="reviews")
    product = relationship("Product", back_populates="reviews")
    
    def get_id_column(self):
        """Get ID column for CRUD operations."""
        return Review.review_id


class Event(Base):
    """Event model for database."""
    __tablename__ = "events"
    __table_args__ = {"schema": db_config['schema']}

    event_id = Column(Integer, primary_key=True)
    user_id = Column(String(255), ForeignKey(f"{db_config['schema']}.users.user_id"))
    product_id = Column(String(255), ForeignKey(f"{db_config['schema']}.products.product_id"))
    event_type = Column(String(50))
    event_timestamp = Column(DateTime, default=func.now())
    session_id = Column(String(255))
    event_data = Column(JSONB)  # Renamed from event_metadata to avoid conflict

    # Relationships
    user = relationship("User", back_populates="events")
    product = relationship("Product", back_populates="events")
    
    def get_id_column(self):
        """Get ID column for CRUD operations."""
        return Event.event_id


class Recommendation(Base):
    """Recommendation model for database."""
    __tablename__ = "recommendations"
    __table_args__ = {"schema": db_config['schema']}

    recommendation_id = Column(Integer, primary_key=True)
    user_id = Column(String(255), ForeignKey(f"{db_config['schema']}.users.user_id"))
    product_id = Column(String(255), ForeignKey(f"{db_config['schema']}.products.product_id"))
    score = Column(Float)
    model_version = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    served_at = Column(DateTime)
    clicked = Column(Boolean, default=False)
    purchased = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="recommendations")
    product = relationship("Product", back_populates="recommendations")
    
    def get_id_column(self):
        """Get ID column for CRUD operations."""
        return Recommendation.recommendation_id


class ModelVersion(Base):
    """Model version for database."""
    __tablename__ = "model_versions"
    __table_args__ = {"schema": db_config['schema']}

    model_version_id = Column(Integer, primary_key=True)
    model_name = Column(String(255))
    model_version = Column(String(255))
    artifact_path = Column(String(1000))
    training_time = Column(Float)
    model_params = Column(JSONB)  # Renamed from model_parameters to be shorter
    model_metrics = Column(JSONB)  # Renamed from metrics to avoid conflict
    is_production = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    control_experiments = relationship("Experiment", foreign_keys="Experiment.control_model_version_id", back_populates="control_model")
    treatment_experiments = relationship("Experiment", foreign_keys="Experiment.treatment_model_version_id", back_populates="treatment_model")
    
    def get_id_column(self):
        """Get ID column for CRUD operations."""
        return ModelVersion.model_version_id


class Experiment(Base):
    """Experiment model for database."""
    __tablename__ = "experiments"
    __table_args__ = {"schema": db_config['schema']}

    experiment_id = Column(Integer, primary_key=True)
    name = Column(String(255))
    description = Column(Text)
    start_date = Column(DateTime, default=func.now())
    end_date = Column(DateTime)
    status = Column(String(50))
    control_model_version_id = Column(Integer, ForeignKey(f"{db_config['schema']}.model_versions.model_version_id"))
    treatment_model_version_id = Column(Integer, ForeignKey(f"{db_config['schema']}.model_versions.model_version_id"))
    experiment_metrics = Column(JSONB)  # Renamed from metrics to avoid conflict
    results = Column(JSONB)

    # Relationships
    control_model = relationship("ModelVersion", foreign_keys=[control_model_version_id], back_populates="control_experiments")
    treatment_model = relationship("ModelVersion", foreign_keys=[treatment_model_version_id], back_populates="treatment_experiments")
    
    def get_id_column(self):
        """Get ID column for CRUD operations."""
        return Experiment.experiment_id


# Database session management
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# CRUD operations
class CRUDBase:
    """Base class for CRUD operations."""
    
    def __init__(self, model):
        self.model = model
    
    def get(self, db, id_value):
        """Get a record by ID."""
        return db.query(self.model).filter(self.model.get_id_column() == id_value).first()
    
    def get_multi(self, db, skip=0, limit=100):
        """Get multiple records."""
        return db.query(self.model).offset(skip).limit(limit).all()
    
    def create(self, db, obj_data):
        """Create a new record."""
        db_obj = self.model(**obj_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def update(self, db, id_value, obj_data):
        """Update a record."""
        db_obj = self.get(db, id_value)
        if db_obj:
            for key, value in obj_data.items():
                setattr(db_obj, key, value)
            db.commit()
            db.refresh(db_obj)
        return db_obj
    
    def delete(self, db, id_value):
        """Delete a record."""
        db_obj = self.get(db, id_value)
        if db_obj:
            db.delete(db_obj)
            db.commit()
            return True
        return False


# Initialize database
def init_db():
    """Initialize database by creating tables if they don't exist."""
    logger.info("Creating database tables...")
    # Create schema if it doesn't exist
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {db_config['schema']}"))
        conn.commit()
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize database
    init_db() 