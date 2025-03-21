import os
import logging
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

class MLFlowConfig:
    """Configuration and utility functions for MLFlow integration."""
    
    def __init__(self, tracking_uri=None, experiment_name="amazon-recommendation-engine"):
        """Initialize MLFlow configuration.
        
        Args:
            tracking_uri: MLFlow tracking server URI
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.experiment_name = experiment_name
        self.client = None
        self.experiment_id = None
    
    def initialize(self):
        """Initialize MLFlow client and create experiment if not exists."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create MLflow client
            self.client = MlflowClient()
            
            # Create experiment if it doesn't exist
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
            else:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            
            # Set experiment as active
            mlflow.set_experiment(self.experiment_name)
            
            logger.info(f"MLFlow initialized with experiment: {self.experiment_name} (ID: {self.experiment_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MLFlow: {e}")
            return False
    
    def log_model(self, model, model_name, params=None, metrics=None):
        """Log a model to MLFlow.
        
        Args:
            model: The model object to log
            model_name: Name of the model
            params: Dictionary of model parameters
            metrics: Dictionary of model metrics
            
        Returns:
            str: Run ID if successful, None otherwise
        """
        try:
            with mlflow.start_run() as run:
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log metrics
                if metrics:
                    mlflow.log_metrics(metrics)
                
                # Log model
                if hasattr(model, 'to_dict'):
                    # LightGBM, XGBoost, etc.
                    mlflow.sklearn.log_model(model, model_name)
                elif hasattr(model, 'state_dict'):
                    # PyTorch models
                    mlflow.pytorch.log_model(model, model_name)
                else:
                    # Generic Python objects
                    mlflow.pyfunc.log_model(model, model_name)
                
                logger.info(f"Logged model {model_name} to MLFlow run {run.info.run_id}")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Failed to log model {model_name} to MLFlow: {e}")
            return None
    
    def load_model(self, model_name, version=None, stage=None):
        """Load a model from MLFlow.
        
        Args:
            model_name: Name of the model
            version: Optional version of the model
            stage: Optional stage of the model (Staging, Production, etc.)
            
        Returns:
            object: Loaded model if successful, None otherwise
        """
        try:
            # Construct URI
            if version:
                uri = f"models:/{model_name}/{version}"
            elif stage:
                uri = f"models:/{model_name}/{stage}"
            else:
                # Default to latest version
                uri = f"models:/{model_name}/latest"
            
            # Load model based on type
            try:
                model = mlflow.pytorch.load_model(uri)
                logger.info(f"Loaded PyTorch model {model_name}")
                return model
            except:
                try:
                    model = mlflow.sklearn.load_model(uri)
                    logger.info(f"Loaded scikit-learn model {model_name}")
                    return model
                except:
                    model = mlflow.pyfunc.load_model(uri)
                    logger.info(f"Loaded generic model {model_name}")
                    return model
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name} from MLFlow: {e}")
            return None
    
    def register_model(self, run_id, model_name, stage=None):
        """Register a model in the MLFlow model registry.
        
        Args:
            run_id: ID of the run that produced the model
            model_name: Name to give the model in the registry
            stage: Optional stage to assign to the model version
            
        Returns:
            int: Version number if successful, None otherwise
        """
        try:
            result = mlflow.register_model(
                f"runs:/{run_id}/{model_name}",
                model_name
            )
            
            # Set stage if provided
            if stage and self.client:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=result.version,
                    stage=stage
                )
            
            logger.info(f"Registered model {model_name} version {result.version}")
            return result.version
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            return None
    
    def log_metrics(self, model_name, metrics):
        """Log metrics for a model.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics to log
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with mlflow.start_run() as run:
                # Log model name as a tag
                mlflow.set_tag("model_name", model_name)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                logger.debug(f"Logged metrics for {model_name}: {metrics}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to log metrics for {model_name}: {e}")
            return False 