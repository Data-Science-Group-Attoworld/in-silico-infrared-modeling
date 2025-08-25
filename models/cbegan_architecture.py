import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, noise_dim, output_dim, n_conditions, pos_embedding_dim):
        super(Decoder, self).__init__()
        self.dense_1 = nn.Linear(noise_dim + n_conditions * pos_embedding_dim, 256)
        self.dense_same = nn.Linear(256, 256)
        self.dense_2 = nn.Linear(256, 384)
        self.dense_out = nn.Linear(384, output_dim)
        self.elu = nn.ELU()

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        x = self.elu(self.dense_1(x))
        x = self.elu(self.dense_same(x))
        x = self.elu(self.dense_2(x))
        x = self.dense_out(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, noise_dim):
        super(Encoder, self).__init__()
        self.dense_1 = nn.Linear(input_dim, 384)
        self.dense_2 = nn.Linear(384, 256)
        self.dense_same = nn.Linear(256, 256)
        self.dense_out = nn.Linear(256, noise_dim)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.elu(self.dense_1(x))
        x = self.elu(self.dense_2(x))
        x = self.elu(self.dense_same(x))
        x = self.dense_out(x)
        return x


class DiscriminatorModel(nn.Module):
    def __init__(self, input_dim, noise_dim, n_conditions, pos_embedding_dim):
        super(DiscriminatorModel, self).__init__()
        self.encoder = Encoder(input_dim, noise_dim)
        self.decoder = Decoder(noise_dim, input_dim, n_conditions, pos_embedding_dim)

    def forward(self, x, c):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, c)
        return decoded


class GeneratorModel(nn.Module):
    def __init__(self, noise_dim, output_dim, n_conditions, pos_embedding_dim):
        super(GeneratorModel, self).__init__()
        self.decoder = Decoder(noise_dim, output_dim, n_conditions, pos_embedding_dim)

    def forward(self, x, c):
        return self.decoder(x, c)
