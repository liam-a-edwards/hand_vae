import numpy as np

from utils.tensor import Tensor


class Module:
    def parameters(self):
        params = []
        for v in vars(self).values():
            params.extend(self._collect(v))
        return params

    def _collect(self, obj):
        if isinstance(obj, Tensor):
            return [obj]
        elif isinstance(obj, Module):
            return obj.parameters()
        elif isinstance(obj, (list, tuple)):
            params = []
            for item in obj:
                params.extend(self._collect(item))
            return params
        else:
            return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        out = self.forward(*inputs)
        return out


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weights = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        )
        self.biases = Tensor(np.zeros(out_features))

    def forward(self, x):
        return x @ self.weights + self.biases


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Encoder(Module):
    def __init__(self, input_size, hidden_size, latent_size):
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2_mean = Linear(hidden_size, latent_size)
        self.fc2_logvar = Linear(hidden_size, latent_size)
        self.relu = ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar


class Decoder(Module):
    def __init__(self, input_size, hidden_size, latent_size):
        self.fc1 = Linear(latent_size, hidden_size)
        self.fc2 = Linear(hidden_size, input_size)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        out = self.sigmoid(self.fc2(h))
        return out


class VAE(Module):
    def __init__(self, input_size, hidden_size, latent_size):
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(input_size, hidden_size, latent_size)

    def reparameterize(self, mean, logvar):
        eps = np.random.randn(*mean.data.shape)
        z = mean + (0.5 * logvar).exp() * eps
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, logvar
