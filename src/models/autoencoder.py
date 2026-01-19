"""
Autoencoder-based Anomaly Detection.
LSTM and Transformer autoencoders for sequence anomaly detection.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.types import ModelType, ContributingFeature
from src.utils import get_logger
from .base import BaseAnomalyDetector

logger = get_logger(__name__)


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for sequence reconstruction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_fc = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent representation."""
        _, (h_n, _) = self.encoder(x)
        # Use last layer's hidden state
        h = h_n[-1]
        z = self.encoder_fc(h)
        return z
    
    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent representation back to sequence."""
        h = self.decoder_fc(z)
        # Repeat for sequence length
        h = h.unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.decoder(h)
        reconstruction = self.output_fc(output)
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and latent."""
        z = self.encode(x)
        reconstruction = self.decode(z, x.size(1))
        return reconstruction, z


class TransformerAutoencoder(nn.Module):
    """Transformer-based autoencoder for sequence reconstruction."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 128
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_fc = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_fc = nn.Linear(d_model, input_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        batch_size, seq_len, _ = x.size()
        
        # Input projection + positional encoding
        x_proj = self.input_fc(x)
        x_proj = x_proj + self.pos_encoding[:, :seq_len, :]
        
        # Encode
        memory = self.encoder(x_proj)
        
        # Get latent (mean pooling)
        z = memory.mean(dim=1)
        
        # Decode
        output = self.decoder(x_proj, memory)
        
        # Project to output
        reconstruction = self.output_fc(output)
        
        return reconstruction, z


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """
    LSTM Autoencoder based anomaly detection.
    
    Learns to reconstruct normal sequences. High reconstruction error
    indicates anomaly.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        threshold: float = 0.5,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize LSTM Autoencoder detector.
        
        Args:
            sequence_length: Input sequence length
            hidden_dim: LSTM hidden dimension
            latent_dim: Latent space dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            threshold: Anomaly threshold
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        super().__init__(
            model_type=ModelType.LSTM_AUTOENCODER,
            threshold=threshold,
            **kwargs
        )
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self._model: Optional[LSTMAutoencoder] = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._threshold_mse: float = 0.0
    
    def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """Prepare sequences from flat input."""
        if len(X.shape) == 2:
            # Already in sequence form
            return X
        
        # Create sequences from 1D data
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "LSTMAutoencoderDetector":
        """
        Fit the LSTM Autoencoder.
        
        Args:
            X: Training data - can be (n_samples, seq_len, features) or (n_samples, features)
            y: Ignored
            feature_names: Feature names
            
        Returns:
            Self
        """
        X = np.atleast_2d(X)
        
        # Prepare sequences if needed
        if len(X.shape) == 2:
            # Reshape to sequences (n_sequences, seq_len, features)
            n_features = X.shape[1]
            X = X.reshape(-1, self.sequence_length, n_features)
        
        n_samples, seq_len, n_features = X.shape
        
        # Normalize
        X_flat = X.reshape(-1, n_features)
        self._scaler_mean = np.mean(X_flat, axis=0)
        self._scaler_std = np.std(X_flat, axis=0)
        self._scaler_std = np.maximum(self._scaler_std, 1e-10)
        
        X_normalized = (X - self._scaler_mean) / self._scaler_std
        
        # Create model
        self._model = LSTMAutoencoder(
            input_dim=n_features,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        
        # Create dataloader
        X_tensor = torch.FloatTensor(X_normalized)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self._model.train()
        training_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                reconstruction, _ = self._model(batch_x)
                loss = criterion(reconstruction, batch_x)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            training_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        # Compute threshold based on training reconstruction errors
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            reconstruction, _ = self._model(X_tensor)
            mse = torch.mean((reconstruction - X_tensor) ** 2, dim=(1, 2))
            self._threshold_mse = float(torch.quantile(mse, 0.95))
        
        self._feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self._is_fitted = True
        
        self.metadata.training_samples = n_samples
        self._training_stats = {
            "final_loss": training_losses[-1],
            "threshold_mse": self._threshold_mse,
        }
        
        logger.info(
            "Fitted LSTM Autoencoder",
            samples=n_samples,
            features=n_features,
            epochs=self.epochs,
            final_loss=training_losses[-1]
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores based on reconstruction error.
        
        Args:
            X: Input data (n_samples, seq_len, features) or (n_samples, features)
            
        Returns:
            Anomaly scores (n_samples,) in range [0, 1]
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Detector must be fitted first")
        
        X = np.atleast_2d(X)
        
        # Handle different input shapes
        if len(X.shape) == 2:
            n_features = X.shape[1]
            X = X.reshape(-1, self.sequence_length, n_features)
        
        # Normalize
        X_normalized = (X - self._scaler_mean) / self._scaler_std
        
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            reconstruction, _ = self._model(X_tensor)
            
            # Compute MSE per sample
            mse = torch.mean((reconstruction - X_tensor) ** 2, dim=(1, 2))
            mse = mse.cpu().numpy()
        
        # Normalize to [0, 1] based on training threshold
        scores = 1 / (1 + np.exp(-(mse - self._threshold_mse) / self._threshold_mse))
        
        return scores
    
    def _get_contributing_features(self, x: np.ndarray) -> List[ContributingFeature]:
        """Get features contributing to anomaly based on reconstruction error."""
        if self._model is None:
            return []
        
        x = np.atleast_2d(x)
        if len(x.shape) == 2:
            x = x.reshape(1, self.sequence_length, -1)
        
        x_normalized = (x - self._scaler_mean) / self._scaler_std
        
        self._model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_normalized).to(self.device)
            reconstruction, _ = self._model(x_tensor)
            
            # Per-feature reconstruction error
            feature_errors = torch.mean((reconstruction - x_tensor) ** 2, dim=1).squeeze()
            feature_errors = feature_errors.cpu().numpy()
        
        # Normalize errors
        total_error = feature_errors.sum()
        if total_error > 0:
            importances = feature_errors / total_error
        else:
            importances = np.ones(len(feature_errors)) / len(feature_errors)
        
        contributions = []
        for i, (name, importance) in enumerate(zip(self._feature_names, importances)):
            if importance > 0.05:
                contributions.append(ContributingFeature(
                    name=name,
                    value=float(x[0, -1, i]),
                    importance=float(importance),
                    expected_range=(
                        float(self._scaler_mean[i] - 2 * self._scaler_std[i]),
                        float(self._scaler_mean[i] + 2 * self._scaler_std[i])
                    )
                ))
        
        contributions.sort(key=lambda c: c.importance, reverse=True)
        return contributions[:5]
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "sequence_length": self.sequence_length,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "model_state_dict": self._model.state_dict() if self._model else None,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
            "threshold_mse": self._threshold_mse,
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model state from deserialization."""
        self.sequence_length = state["sequence_length"]
        self.hidden_dim = state["hidden_dim"]
        self.latent_dim = state["latent_dim"]
        self.num_layers = state["num_layers"]
        self.dropout = state["dropout"]
        self._scaler_mean = state["scaler_mean"]
        self._scaler_std = state["scaler_std"]
        self._threshold_mse = state["threshold_mse"]
        
        if state["model_state_dict"] is not None:
            n_features = len(self._scaler_mean)
            self._model = LSTMAutoencoder(
                input_dim=n_features,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            self._model.load_state_dict(state["model_state_dict"])


class TransformerAutoencoderDetector(BaseAnomalyDetector):
    """
    Transformer Autoencoder based anomaly detection.
    
    Uses attention mechanism to capture long-range dependencies.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        threshold: float = 0.5,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize Transformer Autoencoder detector.
        """
        super().__init__(
            model_type=ModelType.TRANSFORMER_AUTOENCODER,
            threshold=threshold,
            **kwargs
        )
        
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self._model: Optional[TransformerAutoencoder] = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._threshold_mse: float = 0.0
        self._attention_weights: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "TransformerAutoencoderDetector":
        """Fit the Transformer Autoencoder."""
        X = np.atleast_2d(X)
        
        if len(X.shape) == 2:
            n_features = X.shape[1]
            X = X.reshape(-1, self.sequence_length, n_features)
        
        n_samples, seq_len, n_features = X.shape
        
        # Normalize
        X_flat = X.reshape(-1, n_features)
        self._scaler_mean = np.mean(X_flat, axis=0)
        self._scaler_std = np.std(X_flat, axis=0)
        self._scaler_std = np.maximum(self._scaler_std, 1e-10)
        
        X_normalized = (X - self._scaler_mean) / self._scaler_std
        
        # Create model
        self._model = TransformerAutoencoder(
            input_dim=n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_len=seq_len
        ).to(self.device)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        
        X_tensor = torch.FloatTensor(X_normalized)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                reconstruction, _ = self._model(batch_x)
                loss = criterion(reconstruction, batch_x)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        # Compute threshold
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            reconstruction, _ = self._model(X_tensor)
            mse = torch.mean((reconstruction - X_tensor) ** 2, dim=(1, 2))
            self._threshold_mse = float(torch.quantile(mse, 0.95))
        
        self._feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self._is_fitted = True
        
        self.metadata.training_samples = n_samples
        
        logger.info(
            "Fitted Transformer Autoencoder",
            samples=n_samples,
            features=n_features
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Detector must be fitted first")
        
        X = np.atleast_2d(X)
        
        if len(X.shape) == 2:
            n_features = X.shape[1]
            X = X.reshape(-1, self.sequence_length, n_features)
        
        X_normalized = (X - self._scaler_mean) / self._scaler_std
        
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            reconstruction, _ = self._model(X_tensor)
            mse = torch.mean((reconstruction - X_tensor) ** 2, dim=(1, 2))
            mse = mse.cpu().numpy()
        
        scores = 1 / (1 + np.exp(-(mse - self._threshold_mse) / self._threshold_mse))
        
        return scores
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "model_state_dict": self._model.state_dict() if self._model else None,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
            "threshold_mse": self._threshold_mse,
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model state from deserialization."""
        self.sequence_length = state["sequence_length"]
        self.d_model = state["d_model"]
        self.nhead = state["nhead"]
        self.num_encoder_layers = state["num_encoder_layers"]
        self.num_decoder_layers = state["num_decoder_layers"]
        self.dim_feedforward = state["dim_feedforward"]
        self.dropout = state["dropout"]
        self._scaler_mean = state["scaler_mean"]
        self._scaler_std = state["scaler_std"]
        self._threshold_mse = state["threshold_mse"]
        
        if state["model_state_dict"] is not None:
            n_features = len(self._scaler_mean)
            self._model = TransformerAutoencoder(
                input_dim=n_features,
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                num_decoder_layers=self.num_decoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                max_seq_len=self.sequence_length
            ).to(self.device)
            self._model.load_state_dict(state["model_state_dict"])
