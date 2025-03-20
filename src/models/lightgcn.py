"""
LightGCN model for the Amazon recommendation system.

This module implements the LightGCN (Light Graph Convolutional Network) model,
which is a simplified graph convolutional network specifically designed for 
recommendations.

Reference: He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020).
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
In Proceedings of the 43rd International ACM SIGIR Conference on Research and 
Development in Information Retrieval (pp. 639-648).
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from src.models.base_model import MatrixFactorizationBase

# Set up logging
logger = logging.getLogger(__name__)

# Check if PyTorch and PyTorch Geometric are available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    TORCH_AVAILABLE = True
    
    try:
        import torch_geometric
        from torch_geometric.nn.conv import MessagePassing
        from torch_geometric.utils import add_self_loops, degree
        
        TORCH_GEOMETRIC_AVAILABLE = True
    except ImportError:
        logger.warning("PyTorch Geometric not available. Using simplified LightGCN implementation.")
        TORCH_GEOMETRIC_AVAILABLE = False
        
except ImportError:
    logger.warning("PyTorch not available. LightGCN model will not work.")
    TORCH_AVAILABLE = False
    TORCH_GEOMETRIC_AVAILABLE = False


class LightGCNLayer(nn.Module):
    """
    Implementation of a single LightGCN layer.
    
    This layer performs the light graph convolution operation as described in the
    LightGCN paper. It simplifies the GCN by removing feature transformation and
    nonlinear activation.
    """
    
    def __init__(self):
        """Initialize LightGCN layer."""
        super(LightGCNLayer, self).__init__()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LightGCN layer.
        
        Args:
            x: Input node embeddings [num_nodes, embedding_dim]
            adj: Normalized adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node embeddings [num_nodes, embedding_dim]
        """
        # Simple neighborhood aggregation without self-loops
        return torch.sparse.mm(adj, x)


class LightGCNModel(nn.Module):
    """
    LightGCN model implementation.
    
    This model implements the LightGCN architecture for recommendation.
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, 
                num_layers: int = 3, dropout: float = 0.0):
        """
        Initialize LightGCN model.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Dimension of embeddings
            num_layers: Number of LightGCN layers
            dropout: Dropout probability
        """
        super(LightGCNModel, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Create user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # LightGCN layers
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn_layers.append(LightGCNLayer())
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def forward(self, adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LightGCN.
        
        Args:
            adj_matrix: Normalized adjacency matrix
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Get initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Concatenate user and item embeddings
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        all_emb = self.dropout(all_emb)
        
        # List to store embeddings from each layer
        embs = [all_emb]
        
        # Graph convolution
        for gcn_layer in self.gcn_layers:
            all_emb = gcn_layer(all_emb, adj_matrix)
            all_emb = self.dropout(all_emb)
            embs.append(all_emb)
        
        # Sum embeddings from all layers
        embs = torch.stack(embs, dim=0)
        embs = torch.mean(embs, dim=0)
        
        # Split user and item embeddings
        user_embs, item_embs = torch.split(embs, [self.num_users, self.num_items])
        
        return user_embs, item_embs
    
    def predict(self, user_embs: torch.Tensor, item_embs: torch.Tensor, 
               users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_embs: User embeddings
            item_embs: Item embeddings
            users: User indices
            items: Item indices
            
        Returns:
            Predicted ratings
        """
        user_emb = user_embs[users]
        item_emb = item_embs[items]
        
        # Compute dot product
        ratings = torch.sum(user_emb * item_emb, dim=1)
        
        return ratings


class LightGCNDataset(Dataset):
    """Dataset for LightGCN model."""
    
    def __init__(self, user_indices: List[int], item_indices: List[int], 
                ratings: List[float], num_negatives: int = 1):
        """
        Initialize LightGCN dataset.
        
        Args:
            user_indices: List of user indices
            item_indices: List of item indices
            ratings: List of ratings
            num_negatives: Number of negative samples per positive sample
        """
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.ratings = ratings
        self.num_negatives = num_negatives
        
        # Build positive interaction sets
        self.user_items = {}
        for user, item in zip(user_indices, item_indices):
            if user not in self.user_items:
                self.user_items[user] = set()
            self.user_items[user].add(item)
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.user_indices) * (1 + self.num_negatives)
    
    def __getitem__(self, idx: int) -> Tuple[int, int, float]:
        """
        Get an item by index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (user_idx, item_idx, rating)
        """
        # Positive sample
        if idx < len(self.user_indices):
            return self.user_indices[idx], self.item_indices[idx], self.ratings[idx]
        
        # Negative sample
        pos_idx = idx % len(self.user_indices)
        user = self.user_indices[pos_idx]
        
        # Get positive items for user
        pos_items = self.user_items.get(user, set())
        
        # Randomly select an item until we find a negative sample
        num_items = max(self.item_indices) + 1
        while True:
            negative_item = np.random.randint(0, num_items)
            if negative_item not in pos_items:
                return user, negative_item, 0.0


class LightGCN(MatrixFactorizationBase):
    """
    LightGCN model for recommendation.
    
    This class implements the LightGCN (Light Graph Convolutional Network) model,
    which is a simplified graph convolutional network specifically designed for
    recommendations.
    """
    
    def __init__(self, name: str = "LightGCN", version: str = "0.1.0", 
                 embedding_dim: int = 64, num_layers: int = 3,
                 learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 batch_size: int = 1024, epochs: int = 100, 
                 negative_samples: int = 1, early_stopping_patience: int = 10,
                 lambda_reg: float = 1e-6, dropout: float = 0.1,
                 device: str = None):
        """
        Initialize LightGCN model.
        
        Args:
            name: Name of the model
            version: Version of the model
            embedding_dim: Dimension of embeddings
            num_layers: Number of LightGCN layers
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            batch_size: Batch size for training
            epochs: Number of epochs for training
            negative_samples: Number of negative samples per positive sample
            early_stopping_patience: Patience for early stopping
            lambda_reg: Regularization parameter
            dropout: Dropout probability
            device: Device to run model on ('cpu' or 'cuda')
        """
        super().__init__(name, version)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LightGCN")
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.early_stopping_patience = early_stopping_patience
        self.lambda_reg = lambda_reg
        self.dropout = dropout
        
        # Set device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Model will be initialized during training
        self.model = None
        self.optimizer = None
        
        # Set hyperparameters
        self.set_hyperparameters(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            negative_samples=negative_samples,
            lambda_reg=lambda_reg,
            dropout=dropout
        )
    
    def _initialize_model(self, num_users: int, num_items: int) -> None:
        """
        Initialize PyTorch model.
        
        Args:
            num_users: Number of users
            num_items: Number of items
        """
        self.model = LightGCNModel(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _create_adjacency_matrix(self, user_indices, item_indices):
        """Create the adjacency matrix for the bipartite graph."""
        num_users = len(self.user_map)
        num_items = len(self.product_map)
        
        # Create user-item interaction matrix
        adj_matrix = csr_matrix((np.ones(len(user_indices)), 
                               (user_indices, item_indices)), 
                              shape=(num_users, num_items))
        
        # Create bipartite adjacency matrix for GCN
        # [0, R]
        # [R^T, 0]
        adj = sp.bmat([[None, adj_matrix], 
                      [adj_matrix.T, None]], format='csr')
        
        # Compute degree matrix
        degrees = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # Compute normalized adjacency
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        # Convert to PyTorch sparse tensor
        norm_adj = norm_adj.tocoo()
        indices = torch.LongTensor([norm_adj.row, norm_adj.col])
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        # Move to device
        adj_tensor = torch.sparse.FloatTensor(indices, values, shape).to(self.device)
        
        return adj_tensor

    def train(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train the model."""
        logger.info(f"Training {self.name} model")
        
        # Check for required columns
        required_columns = ['user_id', 'product_id', 'rating']
        for col in required_columns:
            if col not in train_data.columns:
                raise ValueError(f"Required column {col} not in training data")
        
        # Create ID mappings
        self._create_id_mappings(train_data)
        
        # Map external IDs to internal indices
        user_indices = [self.user_map[uid] for uid in train_data['user_id']]
        item_indices = [self.product_map[pid] for pid in train_data['product_id']]
        ratings = train_data['rating'].values.astype(np.float32)
        
        # Normalize ratings to [0, 1]
        ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-10)
        
        # Create adjacency matrix
        adj_matrix = self._create_adjacency_matrix(user_indices, item_indices)
        
        # Create dataset
        dataset = LightGCNDataset(user_indices, item_indices, ratings, self.negative_samples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        # Initialize model
        num_users = len(self.user_map)
        num_items = len(self.product_map)
        logger.info(f"Initializing model with {num_users} users and {num_items} items")
        self._initialize_model(num_users, num_items)
        
        # Training loop
        start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        logger.info(f"Starting training for {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_user, batch_item, batch_rating in dataloader:
                # Forward pass
                batch_user = batch_user.to(self.device)
                batch_item = batch_item.to(self.device)
                batch_rating = batch_rating.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Get user and item embeddings through the GCN
                user_embs, item_embs = self.model(adj_matrix)
                
                # Get embeddings for current batch
                batch_user_embs = user_embs[batch_user]
                batch_item_embs = item_embs[batch_item]
                
                # Predict ratings
                pred_ratings = torch.sum(batch_user_embs * batch_item_embs, dim=1)
                
                # Calculate BPR loss
                pos_mask = batch_rating > 0
                pos_indices = pos_mask.nonzero(as_tuple=True)[0]
                neg_indices = (~pos_mask).nonzero(as_tuple=True)[0]
                
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    # Create pairs of positive and negative examples
                    pos_users = batch_user[pos_indices]
                    pos_items = batch_item[pos_indices]
                    neg_users = batch_user[neg_indices]
                    neg_items = batch_item[neg_indices]
                    
                    # Ensure equal number of positive and negative examples
                    min_size = min(len(pos_indices), len(neg_indices))
                    pos_users = pos_users[:min_size]
                    pos_items = pos_items[:min_size]
                    neg_users = neg_users[:min_size]
                    neg_items = neg_items[:min_size]
                    
                    # Get predictions
                    pos_ratings = torch.sum(user_embs[pos_users] * item_embs[pos_items], dim=1)
                    neg_ratings = torch.sum(user_embs[neg_users] * item_embs[neg_items], dim=1)
                    
                    # Debug info
                    logger.debug(f"Pos ratings shape: {pos_ratings.shape}, Neg ratings shape: {neg_ratings.shape}")
                    
                    # BPR loss
                    loss = -torch.mean(F.logsigmoid(pos_ratings - neg_ratings))
                    
                    # L2 regularization
                    l2_reg = 0
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param)
                    
                    loss += self.lambda_reg * l2_reg
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
            
            train_loss /= len(dataloader)
            train_losses.append(train_loss)
            
            # Validation
            if validation_data is not None:
                self.model.eval()
                with torch.no_grad():
                    # Get embeddings
                    user_embs, item_embs = self.model(adj_matrix)
                    
                    # Map validation data
                    val_user_indices = []
                    val_item_indices = []
                    val_ratings = []
                    
                    for _, row in validation_data.iterrows():
                        user_idx = self.user_map.get(row['user_id'])
                        item_idx = self.product_map.get(row['product_id'])
                        
                        if user_idx is not None and item_idx is not None:
                            val_user_indices.append(user_idx)
                            val_item_indices.append(item_idx)
                            val_ratings.append(row['rating'])
                    
                    # Convert to tensors
                    val_users = torch.LongTensor(val_user_indices).to(self.device)
                    val_items = torch.LongTensor(val_item_indices).to(self.device)
                    val_ratings = torch.FloatTensor(val_ratings).to(self.device)
                    
                    # Normalize validation ratings
                    val_ratings = (val_ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-10)
                    
                    # Predict ratings
                    val_user_embs = user_embs[val_users]
                    val_item_embs = item_embs[val_items]
                    pred_ratings = torch.sum(val_user_embs * val_item_embs, dim=1)
                    
                    # Calculate loss
                    val_loss = F.mse_loss(pred_ratings, val_ratings).item()
                    val_losses.append(val_loss)
                    
                    # Check for early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # Save embeddings
                        self.user_factors = user_embs.cpu().detach().numpy()
                        self.item_factors = item_embs.cpu().detach().numpy()
                        
                        logger.info(f"New best validation loss: {best_val_loss:.6f}")
                    else:
                        patience_counter += 1
                        logger.info(f"Early stopping patience: {patience_counter}/{self.early_stopping_patience}")
                        
                        if patience_counter >= self.early_stopping_patience:
                            logger.info(f"Early stopping triggered after {epoch+1} epochs")
                            break
            else:
                # No validation data, just save the embeddings from the current epoch
                with torch.no_grad():
                    user_embs, item_embs = self.model(adj_matrix)
                    self.user_factors = user_embs.cpu().detach().numpy()
                    self.item_factors = item_embs.cpu().detach().numpy()
            
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}" + 
                       (f", Val Loss: {val_losses[-1]:.6f}" if val_losses else ""))
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Update metadata
        self.metadata['trained'] = True
        self.metadata['training_time'] = training_time
        
        # Metrics
        metrics = {
            'train_loss': train_losses[-1],
            'training_time': training_time,
            'num_epochs': epoch + 1,
            'num_users': num_users,
            'num_items': num_items,
        }
        
        if val_losses:
            metrics['val_loss'] = val_losses[-1]
            metrics['best_val_loss'] = best_val_loss
        
        self.metrics = metrics
        
        return metrics
    
    def predict(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: ID of the user
            n: Number of recommendations to generate
            
        Returns:
            List of tuples containing (product_id, score)
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model has not been trained yet")
        
        user_idx = self._map_user_id(user_id)
        if user_idx is None:
            logger.warning(f"User {user_id} not found in model")
            return []
        
        # Get user vector
        user_vector = self.user_factors[user_idx]
        
        # Calculate scores for all items
        scores = np.dot(self.item_factors, user_vector)
        
        # Get top N items
        top_indices = np.argsort(scores)[::-1][:n]
        
        # Map to product IDs and scores
        recommendations = [
            (self.reverse_product_map[idx], float(scores[idx]))
            for idx in top_indices
        ]
        
        return recommendations
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data DataFrame
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model has not been trained yet")
        
        logger.info(f"Evaluating {self.name} model on {len(test_data)} instances")
        
        # Map user and item IDs to internal indices
        mapped_data = []
        
        for _, row in test_data.iterrows():
            user_idx = self.user_map.get(row['user_id'])
            item_idx = self.product_map.get(row['product_id'])
            
            # Skip if user or item not in training data
            if user_idx is None or item_idx is None:
                continue
            
            mapped_data.append({
                'user_idx': user_idx,
                'item_idx': item_idx,
                'rating': row['rating']
            })
        
        # Convert to DataFrame
        mapped_df = pd.DataFrame(mapped_data)
        
        if len(mapped_df) == 0:
            logger.warning("No test instances could be mapped to training data")
            return {'rmse': float('inf'), 'mae': float('inf')}
        
        # Predict ratings
        predicted_ratings = []
        
        for _, row in mapped_df.iterrows():
            user_vector = self.user_factors[row['user_idx']]
            item_vector = self.item_factors[row['item_idx']]
            pred = float(np.dot(user_vector, item_vector))
            predicted_ratings.append(pred)
        
        # Calculate metrics
        actual_ratings = mapped_df['rating'].values
        predicted_ratings = np.array(predicted_ratings)
        
        # Scale predictions to original rating scale
        min_rating = test_data['rating'].min()
        max_rating = test_data['rating'].max()
        predicted_ratings = predicted_ratings * (max_rating - min_rating) + min_rating
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((predicted_ratings - actual_ratings) ** 2))
        
        # Calculate MAE
        mae = np.mean(np.abs(predicted_ratings - actual_ratings))
        
        # Calculate precision and recall at k
        k_values = [5, 10, 20]
        precision_at_k = {}
        recall_at_k = {}
        
        # Group by user to calculate precision and recall
        for k in k_values:
            precision_sum = 0
            recall_sum = 0
            user_count = 0
            
            for user_id in self.user_map.keys():
                user_idx = self.user_map[user_id]
                
                # Get ground truth items for this user
                relevant_items = set(
                    row['item_idx'] for _, row in mapped_df.iterrows() 
                    if row['user_idx'] == user_idx and row['rating'] >= 4  # Consider ratings >= 4 as relevant
                )
                
                if len(relevant_items) == 0:
                    continue
                
                # Get recommendations for this user
                recs = self.predict(user_id, n=k)
                recommended_items = set(
                    self.product_map[prod_id] for prod_id, _ in recs
                )
                
                # Calculate precision and recall
                if len(recommended_items) > 0:
                    relevant_and_recommended = len(relevant_items.intersection(recommended_items))
                    precision = relevant_and_recommended / len(recommended_items)
                    recall = relevant_and_recommended / len(relevant_items)
                    
                    precision_sum += precision
                    recall_sum += recall
                    user_count += 1
            
            # Calculate average precision and recall
            if user_count > 0:
                precision_at_k[f'precision@{k}'] = precision_sum / user_count
                recall_at_k[f'recall@{k}'] = recall_sum / user_count
            else:
                precision_at_k[f'precision@{k}'] = 0.0
                recall_at_k[f'recall@{k}'] = 0.0
        
        # Combine all metrics
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'num_test_instances': len(mapped_df),
            **precision_at_k,
            **recall_at_k
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def _get_model_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.
        
        Returns:
            Dictionary containing model data
        """
        model_data = super()._get_model_data()
        
        # Add PyTorch model state dict if available
        if self.model is not None:
            model_data['torch_state_dict'] = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in self.model.state_dict().items()
            }
        
        return model_data
    
    def _set_model_data(self, data: Dict[str, Any]) -> None:
        """
        Set model-specific data after loading.
        
        Args:
            data: Dictionary containing model data
        """
        super()._set_model_data(data)
        
        # Recreate PyTorch model if state dict is available
        if 'torch_state_dict' in data and len(self.user_map) > 0 and len(self.product_map) > 0:
            self._initialize_model(len(self.user_map), len(self.product_map))
            
            # Convert numpy arrays back to torch tensors
            state_dict = {
                k: torch.tensor(v) if isinstance(v, np.ndarray) else v
                for k, v in data['torch_state_dict'].items()
            }
            
            self.model.load_state_dict(state_dict)
            self.model.eval()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required to run this example")
        exit(1)
    
    # Create synthetic data
    num_users = 100
    num_items = 50
    num_samples = 1000
    
    np.random.seed(42)
    
    user_ids = [f"user_{i}" for i in range(num_users)]
    item_ids = [f"item_{i}" for i in range(num_items)]
    
    # Generate random interactions
    train_data = []
    for _ in range(num_samples):
        user_id = np.random.choice(user_ids)
        item_id = np.random.choice(item_ids)
        rating = np.random.randint(1, 6)  # 1-5 rating
        
        train_data.append({
            'user_id': user_id,
            'product_id': item_id,
            'rating': rating
        })
    
    # Convert to DataFrame
    train_df = pd.DataFrame(train_data)
    
    # Create and train LightGCN model
    lightgcn_model = LightGCN(
        embedding_dim=32,
        num_layers=2,
        epochs=5  # Small number of epochs for example
    )
    
    # Train model
    metrics = lightgcn_model.train(train_df)
    
    # Make predictions
    user_id = user_ids[0]
    recommendations = lightgcn_model.predict(user_id, n=5)
    
    logger.info(f"Top 5 recommendations for {user_id}: {recommendations}") 