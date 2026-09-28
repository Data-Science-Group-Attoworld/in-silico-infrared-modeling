import torch
import torch.nn as nn
import math
from utils.utils import position_encoding

class CfgDiffusion(nn.Module):
    def __init__(
        self, 
        input_features, 
        n_conditions, 
        model_type='MLP',
        hidden_dim=1024,
        pos_embedding_dim=20,
        timestep_embedding_dim=20,
        n_layer=4, 
        n_timesteps=64, 
        p_uncond=0.15, 
        activation='silu',
    ):
        super().__init__()
        self.input_features = input_features
        self.n_conditions = n_conditions
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.timestep_embedding_dim = timestep_embedding_dim
        self.n_layer = n_layer
        self.n_timesteps=n_timesteps
        self.p_uncond = p_uncond
        self.activation = activation

        condition_dim = n_conditions*pos_embedding_dim
        
        # Time embedding (sinusoidal → linear projection)
        self.time_embed = nn.Sequential(
            PositionalEmbeddingModule(self.timestep_embedding_dim, base=n_timesteps*2),   # *2 is arbitrary, just something that results in smooth coverage
            nn.Linear(self.timestep_embedding_dim, self.timestep_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.timestep_embedding_dim, self.timestep_embedding_dim)
        )
        
        # Condition embedding (continuous demographics)
        self.cond_proj = nn.Sequential(
            nn.Linear(condition_dim, condition_dim), 
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        self.null_cond = nn.Parameter(torch.zeros(1, condition_dim))

        if self.model_type == 'MLP':
            # MLP backbone
            dims = [input_features + condition_dim + self.timestep_embedding_dim] + [hidden_dim] * n_layer + [input_features]
            layers = []
            for i, (d_in, d_out) in enumerate(zip(dims, dims[1:])):
                layers.append(nn.Linear(d_in, d_out))
                if i < len(dims) - 2:
                    act = self._get_activation(self.activation)
                    layers += [nn.LayerNorm(d_out), act]
            self.net = nn.Sequential(*layers)

        elif self.model_type == 'ResMLP':
            # Residual MLP backbone
            input_dim = input_features + condition_dim + self.timestep_embedding_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                *[ResMLPBlock(hidden_dim, activation=activation) for _ in range(n_layer)],
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, input_features),
            )

        elif self.model_type == 'FiLMResMLP':
            # FiLM Residual MLP backbone
            self.input_proj = nn.Linear(input_features, hidden_dim)
            self.blocks = nn.ModuleList([
                FiLMResMLPBlock(hidden_dim, condition_dim + self.timestep_embedding_dim, activation=activation)
                for _ in range(n_layer)
            ])
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, input_features)
    
    def forward(self, x, y, t, drop_labels=None):
        t_emb = self.time_embed(t)
        y_emb = self.cond_proj(y)
    
        if drop_labels is not None:
            null = self.null_cond.expand(y_emb.size(0), -1)
            y_emb = torch.where(drop_labels.unsqueeze(1), null, y_emb)

        if self.model_type == 'MLP':
            # MLP
            h = torch.cat([x, t_emb, y_emb], dim=1)
            return self.net(h)

        elif self.model_type == 'ResMLP':
            # Residual MLP
            h = torch.cat([x, t_emb, y_emb], dim=1)
            return self.net(h)

        elif self.model_type == 'FiLMResMLP':
            # FiLM Residual MLP
            cond = torch.cat([t_emb, y_emb], dim=1)
            h = self.input_proj(x)
            for block in self.blocks:
                h = block(h, cond)
            h = self.final_norm(h)
            return self.output_proj(h)

    def forward_uncond(self, x, t):
        B = x.size(0)
        t_emb = self.time_embed(t)
        null = self.null_cond.expand(B, -1)

        if self.model_type == 'MLP':
            # MLP
            h = torch.cat([x, t_emb, null], dim=1)
            return self.net(h)

        elif self.model_type == 'ResMLP':
            # Residual MLP
            h = torch.cat([x, t_emb, null], dim=1)
            return self.net(h)

        elif self.model_type == 'FiLMResMLP':
            # FiLM Residual MLP
            cond = torch.cat([t_emb, null], dim=1)
            h = self.input_proj(x)
            for block in self.blocks:
                h = block(h, cond)
            h = self.final_norm(h)
            return self.output_proj(h)


    def _get_activation(self, activation):
        acts = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "sigmoid": nn.Sigmoid,
        }
        if activation not in acts:
            raise ValueError(f"Unsupported activation function: {activation}")
        return acts[activation]()
        

class PositionalEmbeddingModule(nn.Module):
    """
    Wraps position_encoding as an nn.Module for use inside model architectures.
    Applies sinusoidal positional encoding to a batch of scalar values.
    
    Args:
        embedding_dim: output embedding size (must be even)
        base: frequency base (default 10000, as in Vaswani et al. / DDPM)
              use 0.1 for legacy compatibility with other project models
    """
    def __init__(self, embedding_dim: int, base: float = 10000.0):
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even"
        self.embedding_dim = embedding_dim
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,) or (B, 1) scalar values (integer timesteps or continuous)
        Returns:
            (B, embedding_dim)
        """
        if x.ndim == 1:
            x = x.unsqueeze(1)
        return position_encoding(x, self.embedding_dim, x.device, self.base)



class ResMLPBlock(nn.Module):
    def __init__(self, dim, activation="silu", dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = self._get_activation(activation)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h  # residual connection

    def _get_activation(self, activation):
        acts = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "sigmoid": nn.Sigmoid,
        }
        if activation not in acts:
            raise ValueError(f"Unsupported activation function: {activation}")
        return acts[activation]()




class FiLMResMLPBlock(nn.Module):
    def __init__(self, dim, cond_dim, activation="silu", dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # FiLM modulation parameters
        self.film = nn.Linear(cond_dim, dim * 2)

        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = self._get_activation(activation)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        """
        x:    (B, dim)
        cond: (B, cond_dim)
        """
        h = self.norm(x)

        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = 1 + gamma  # stabilize early training

        h = gamma * h + beta

        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.dropout(h)

        return x + h

    def _get_activation(self, activation):
        acts = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        if activation not in acts:
            raise ValueError(f"Unsupported activation function: {activation}")
        return acts[activation]()