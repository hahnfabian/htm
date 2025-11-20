class Config:
    def __init__(self, epochs=100, learning_rate=1e-3, batch_size=64, model_name=None, sampling_policy="softmax", device="cpu", token_dim=None, latent_dim=None, hidden_dim=None, checkpoint_path="./checkpoints/null.pth"):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_name = model_name
        self.sampling_policy = sampling_policy
        self.device = device
        self.token_dim=token_dim,
        self.latent_dim=latent_dim,
        self.hidden_dim=hidden_dim
        self.checkpoint_path = checkpoint_path