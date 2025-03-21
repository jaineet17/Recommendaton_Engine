#!/usr/bin/env python3
"""
Legacy Model Import Script

This script helps import models from the old flat structure to the new organized structure.
It looks for models in the legacy locations and imports them into the proper versioned structure.

Usage:
    python legacy_model_import.py [--source-dir PATH] [--target-dir PATH] [--version VERSION]
"""

import argparse
import os
import json
import shutil
import pickle
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Import legacy models to the new model structure")
    parser.add_argument("--source-dir", default="data/models/archived/legacy_flat_structure",
                      help="Directory containing legacy model files")
    parser.add_argument("--target-dir", default="data/models",
                      help="Root directory of the new model structure")
    parser.add_argument("--version", default="1.0.0",
                      help="Version to assign to imported models")
    return parser.parse_args()

def load_pickle_safely(filepath):
    """Load a pickle file and extract basic info without full deserialization if possible."""
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None

def extract_model_metadata(model, model_type):
    """Extract metadata from a model object."""
    metadata = {
        "hyperparameters": {}
    }
    
    # Extract common metadata
    if isinstance(model, dict):
        # Extract version info if available
        metadata["version"] = model.get("version", "unknown")
        
        # Extract metrics if available
        if "metrics" in model:
            metadata["metrics"] = model.get("metrics", {})
            
        # Try to extract hyperparameters
        if model_type == "lightgcn" and "params" in model:
            params = model.get("params", {})
            metadata["hyperparameters"] = {
                "embedding_dim": params.get("embedding_dim", 64),
                "n_layers": params.get("n_layers", 3),
                "learning_rate": params.get("learning_rate", 0.001)
            }
        elif model_type == "ncf" and "params" in model:
            params = model.get("params", {})
            metadata["hyperparameters"] = {
                "embedding_dim": params.get("embedding_dim", 32),
                "layers": params.get("layers", [64, 32, 16]),
                "dropout": params.get("dropout", 0.2),
            }
        elif model_type == "matrix_factorization" and "params" in model:
            params = model.get("params", {})
            metadata["hyperparameters"] = {
                "n_factors": params.get("n_factors", 50),
                "reg_user": params.get("reg_user", 0.02),
                "reg_item": params.get("reg_item", 0.02),
            }
        elif model_type == "content_based" and "params" in model:
            params = model.get("params", {})
            metadata["hyperparameters"] = {
                "vectorizer": params.get("vectorizer", "TfidfVectorizer"),
                "min_df": params.get("min_df", 5),
                "max_features": params.get("max_features", 10000),
            }
        elif model_type == "ensemble" and "metrics" in model:
            metrics = model.get("metrics", {})
            metadata["hyperparameters"] = {
                "lightgcn_weight": metrics.get("lightgcn_weight", 0.7),
                "content_weight": metrics.get("content_weight", 0.3),
            }
            if "components" in model:
                metadata["component_models"] = model.get("components", [])
                
    return metadata

def import_legacy_model(source_path, target_dir, model_type, version):
    """Import a legacy model to the new structure."""
    # Determine target directory
    model_dir_map = {
        "lightgcn": "lightgcn",
        "ncf": "ncf",
        "simple_mf": "matrix_factorization",
        "content_based": "content_based",
        "ensemble": "ensemble"
    }
    
    if model_type not in model_dir_map:
        logger.error(f"Unknown model type: {model_type}")
        return False
    
    model_dir = model_dir_map[model_type]
    target_model_dir = os.path.join(target_dir, model_dir)
    
    # Ensure target directory exists
    os.makedirs(target_model_dir, exist_ok=True)
    
    # Load the model to extract metadata
    model = load_pickle_safely(source_path)
    if model is None:
        return False
    
    # Extract metadata
    model_metadata = extract_model_metadata(model, model_type)
    
    # Determine target filename
    if model_type == "simple_mf":
        target_filename = f"mf_model_v{version}.pkl"
    else:
        target_filename = f"{model_type}_model_v{version}.pkl"
    
    target_path = os.path.join(target_model_dir, target_filename)
    
    # Copy the model file
    try:
        shutil.copy2(source_path, target_path)
        logger.info(f"Copied {source_path} to {target_path}")
    except Exception as e:
        logger.error(f"Error copying {source_path} to {target_path}: {e}")
        return False
    
    # Update metadata file
    metadata_path = os.path.join(target_model_dir, "metadata.json")
    
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "model_type": model_dir,
                "versions": [],
                "latest_version": version,
                "production_version": version
            }
        
        # Check if this version already exists
        version_exists = False
        for v in metadata.get("versions", []):
            if v.get("version") == version:
                version_exists = True
                logger.warning(f"Version {version} already exists for {model_type}")
                break
        
        if not version_exists:
            new_version = {
                "version": version,
                "filename": target_filename,
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "is_production": True,
                "description": f"Imported from legacy model structure: {os.path.basename(source_path)}",
                "hyperparameters": model_metadata.get("hyperparameters", {})
            }
            
            # Add component models for ensemble models
            if model_type == "ensemble" and "component_models" in model_metadata:
                new_version["component_models"] = model_metadata["component_models"]
                
            metadata["versions"].append(new_version)
            metadata["latest_version"] = version
            metadata["production_version"] = version
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"Updated metadata for {model_type} version {version}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error updating metadata for {model_type}: {e}")
        return False

def update_registry(target_dir):
    """Update the model registry with imported models."""
    registry_path = os.path.join(target_dir, "registry.json")
    
    try:
        # Load existing registry
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {
                "models": {},
                "model_aliases": {}
            }
        
        # Update last_updated
        registry["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        
        # Scan model directories to update registry
        for model_dir in ["lightgcn", "ncf", "matrix_factorization", "content_based", "ensemble"]:
            model_path = os.path.join(target_dir, model_dir)
            metadata_path = os.path.join(model_path, "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Determine model key
                if model_dir == "matrix_factorization":
                    model_key = "mf"
                elif model_dir == "ensemble":
                    model_key = "hybrid"
                else:
                    model_key = model_dir
                
                # Update registry
                registry["models"][model_key] = {
                    "path": model_dir,
                    "latest_version": metadata.get("latest_version", "1.0.0"),
                    "production_version": metadata.get("production_version", "1.0.0"),
                    "available_versions": [v.get("version") for v in metadata.get("versions", [])]
                }
        
        # Ensure model aliases are set
        if "model_aliases" not in registry:
            registry["model_aliases"] = {
                "default": "lightgcn",
                "collaborative": "lightgcn",
                "neural": "ncf",
                "matrix_factorization": "mf",
                "content": "content_based",
                "ensemble": "hybrid"
            }
        
        # Write updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=4)
            
        logger.info(f"Updated model registry at {registry_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating registry: {e}")
        return False

def main():
    """Main import function."""
    args = parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir
    version = args.version
    
    logger.info(f"Importing legacy models from {source_dir} to {target_dir} with version {version}")
    
    if not os.path.exists(source_dir):
        logger.error(f"Source directory {source_dir} does not exist")
        return
    
    # Map of filename patterns to model types
    model_files = {
        "lightgcn_model.pkl": "lightgcn",
        "ncf_model.pkl": "ncf",
        "simple_mf_model.pkl": "simple_mf",
        "content_based_model.pkl": "content_based",
        "ensemble_model.pkl": "ensemble"
    }
    
    # Import each model type
    successful_imports = 0
    for filename, model_type in model_files.items():
        source_path = os.path.join(source_dir, filename)
        if os.path.exists(source_path):
            logger.info(f"Importing {model_type} model from {source_path}")
            if import_legacy_model(source_path, target_dir, model_type, version):
                successful_imports += 1
        else:
            logger.warning(f"Legacy model file not found: {source_path}")
    
    # Update the registry
    if successful_imports > 0:
        update_registry(target_dir)
        
    logger.info(f"Import completed. Successfully imported {successful_imports} models.")

if __name__ == "__main__":
    main() 