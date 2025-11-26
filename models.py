import torch.nn as nn

class AE(nn.Module):
    def __init__(self, token_dim, latent_dim, hidden_dim):
        super().__init__()

        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.lift = nn.Sequential(
            nn.Linear(self.token_dim, self.latent_dim),
        )
        self.unlift = nn.Sequential(nn.Linear(self.latent_dim, self.token_dim))

        self.encoder = nn.Sequential(
            nn.Linear(2 * self.latent_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(), 
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
            # nn.Tanh(),
        )

        self.apply(self.init_weights)


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.001)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    