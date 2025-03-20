"""
Test script for SQLAlchemy models without connecting to a database.

This script imports the models and performs basic validation without connecting to a database.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)

from sqlalchemy import inspect

# Import the models from the database module
try:
    from src.data.database import (
        Base, Product, User, Review, Event, Recommendation, 
        ModelVersion, Experiment
    )
    print("✅ Successfully imported models")
except ImportError as e:
    print(f"❌ Error importing models: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

# Check model attributes
def check_model(model_class):
    """Check if a model has the required attributes."""
    try:
        # Get the mapper for the model
        mapper = inspect(model_class)
        
        # Print model info
        print(f"\nModel: {model_class.__name__}")
        print("  Tablename:", model_class.__tablename__)
        print("  Primary key(s):", ", ".join(c.name for c in mapper.primary_key))
        
        # Print column info
        print("  Columns:")
        for column in mapper.columns:
            print(f"    - {column.name}: {column.type}")
        
        # Check relationships
        print("  Relationships:")
        for rel in mapper.relationships:
            print(f"    - {rel.key} -> {rel.target.name}")
        
        # Try to call the get_id_column method if it exists
        if hasattr(model_class, 'get_id_column'):
            print(f"  get_id_column method exists: {model_class.get_id_column.__name__}")
        else:
            print("  ❌ get_id_column method is missing")
        
        return True
    except Exception as e:
        print(f"❌ Error checking model {model_class.__name__}: {e}")
        return False

# Check all models
models = [Product, User, Review, Event, Recommendation, ModelVersion, Experiment]
all_valid = True

print("\n--- Checking all models ---")
for model in models:
    if not check_model(model):
        all_valid = False

if all_valid:
    print("\n✅ All models passed basic validation")
    sys.exit(0)
else:
    print("\n❌ One or more models failed validation")
    sys.exit(1) 