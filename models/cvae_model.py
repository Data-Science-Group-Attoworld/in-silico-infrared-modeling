import torch
import torch.nn as nn
from utils.utils import encode_labels


class CondVAE(nn.Module):
    """
    Class to create a Variational Autoencoder with specific attributes.

    Attributes:
        L (int): Number of layers in the encoder/decoder.
        latent_dim (int): Dimension of the latent space.
        activation (nn.Module): Activation function used in the layers.
        input_features (int): Dimension of the input signal.

    """

    def __init__(
        self,
        input_features,
        n_conditions,
        mean,
        sigma,
        n_layer,
        latent_dim,
        activation_function,
        pos_encoding_embedding_dim,
    ):
        super(CondVAE, self).__init__()

        # Extract parameters
        self.n_layer = n_layer
        self.latent_dim = latent_dim
        self.activation = self._get_activation(activation_function)
        self.pos_encoding_embedding_dim = pos_encoding_embedding_dim

        self.input_features = input_features
        self.num_labels = n_conditions

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

        # Hidden dimensions
        hidden_dim = self._calculate_hidden_dimensions()
        self.hidden_dim_encoder = hidden_dim.copy()
        self.hidden_dim_encoder[0] += self.num_labels * self.pos_encoding_embedding_dim
        self.hidden_dim_decoder = hidden_dim.copy()
        self.hidden_dim_decoder[-1] += self.num_labels * self.pos_encoding_embedding_dim

        # Encoder
        encoder_layers = []
        for i in range(self.n_layer):
            encoder_layers.append(
                nn.Linear(
                    in_features=self.hidden_dim_encoder[i],
                    out_features=self.hidden_dim_encoder[i + 1],
                )
            )
            encoder_layers.append(self.activation)
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space layers (no activation for these layers to allow positive and negative mean values and sigma between 0 and 1 for negative logvar)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        # Decoder
        decoder_layers = []
        for i in range(n_layer):
            decoder_layers.append(
                nn.Linear(
                    in_features=self.hidden_dim_decoder[-i - 1],
                    out_features=self.hidden_dim_decoder[-i - 2],
                )
            )
            if (
                i != n_layer - 1
            ):  # No activation in the last decoder layer because it is a reconstruction task
                decoder_layers.append(self.activation)
        self.decoder = nn.Sequential(*decoder_layers)

    def _calculate_hidden_dimensions(self):
        """
        Calculate the dimensions of each layer.

        Returns:
            List[int]: List of dimensions for each layer.
        """
        hidden_dim = [
            round(
                self.input_features
                - i * (self.input_features - self.latent_dim) / (self.n_layer)
            )
            for i in range(self.n_layer + 1)
        ]
        hidden_dim[-1] = self.latent_dim
        return hidden_dim

    def _get_activation(self, activation):
        activations = {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "elu": nn.ELU()}
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        return activations[activation]

    def standard_scale_transform(self, batch):
        return (batch - self.mean) / self.sigma

    def standard_scale_inverse_transform(self, batch):
        return batch * self.sigma + self.mean

    def encode(self, x, labels):
        device = x.device
        label_embedding = encode_labels(labels, self.pos_encoding_embedding_dim, device)
        x = torch.cat((x, label_embedding), dim=1)

        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return encoded, mu, logvar

    def decode(self, latent_representation, labels):
        device = latent_representation.device
        label_embedding = encode_labels(labels, self.pos_encoding_embedding_dim, device)
        latent_representation = torch.cat(
            (latent_representation, label_embedding), dim=1
        )
        return self.decoder(latent_representation)

    def sample(self, mu, logvar):
        """
        Sample from the latent space using reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)  # Numerically stable std calculation
        eps = torch.randn_like(std)  # Random normal noise
        return mu + eps * std

    def forward(self, x, labels):
        encoded, mu, logvar = self.encode(x, labels)
        z = self.sample(mu, logvar)
        decoded = self.decode(z, labels)
        return encoded, decoded, z, mu, logvar
