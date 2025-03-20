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
            model: The model to log
            model_name: Name of the model
            params: Dictionary of parameters
            metrics: Dictionary of metrics
            
        Returns:
            run_id: ID of the MLFlow run
        """
        try:
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log metrics
                if metrics:
                    mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(model, model_name)
                
                # Log additional information
                mlflow.set_tag("model_name", model_name)
                
                # Return run ID
                return run.info.run_id
        except Exception as e:
            logger.error(f"Failed to log model to MLFlow: {e}")
            return None
    
    def log_metrics(self, model_name, metrics):
        """Log metrics for a model.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            
        Returns:
            run_id: ID of the MLFlow run
        """
        try:
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log additional information
                mlflow.set_tag("model_name", model_name)
                
                # Return run ID
                return run.info.run_id
        except Exception as e:
            logger.error(f"Failed to log metrics to MLFlow: {e}")
            return None
    
    def register_model(self, run_id, model_name, stage="None"):
        """Register a model from a run.
        
        Args:
            run_id: ID of the MLFlow run
            model_name: Name of the model
            stage: Stage of the model (None, Staging, Production)
            
        Returns:
            version: Version of the registered model
        """
        try:
            result = mlflow.register_model(
                f"runs:/{run_id}/{model_name}",
                model_name
            )
            
            # Set stage if provided
            if stage != "None":
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=result.version,
                    stage=stage
                )
            
            logger.info(f"Registered model {model_name} version {result.version} in stage {stage}")
            return result.version
        except Exception as e:
            logger.error(f"Failed to register model in MLFlow: {e}")
            return None
    
    def load_model(self, model_name, stage="Production"):
        """Load a model from MLFlow.
        
        Args:
            model_name: Name of the model
            stage: Stage of the model (Production, Staging, None)
            
        Returns:
            model: The loaded model
        """
        try:
            # Construct model URI
            model_uri = f"models:/{model_name}/{stage}"
            
            # Load model
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model {model_name} from stage {stage}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model from MLFlow: {e}")
            # Fallback to loading most recent version
            try:
                logger.info(f"Attempting to load latest version of {model_name}")
                latest_version = self._get_latest_version(model_name)
                if latest_version:
                    model_uri = f"models:/{model_name}/{latest_version}"
                    model = mlflow.sklearn.load_model(model_uri)
                    logger.info(f"Loaded model {model_name} version {latest_version}")
                    return model
            except Exception as inner_e:
                logger.error(f"Failed to load latest model version: {inner_e}")
            return None
    
    def _get_latest_version(self, model_name):
        """Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            version: Latest version of the model
        """
        try:
            # Get all versions of the model
            versions = self.client.get_latest_versions(model_name)
            if versions:
                # Return the highest version number
                return max([int(v.version) for v in versions])
            return None
        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            return None 