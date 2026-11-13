import numpy as np
from tqdm import tqdm

from utils.dataset import load_data
from utils.layers import VAE
from utils.losses import vae_loss
from utils.optim import Adam
from utils.save import save_model
from utils.tensor import Tensor


def main():
    X_train, _y_train, _X_test, _y_test = load_data()
    X = X_train
    config = {"input_size": 784, "hidden_size": 64, "latent_size": 16}
    model = VAE(**config)
    optimizer = Adam(model.parameters())

    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        batch_size = 128
        perm = np.random.permutation(len(X))
        pbar = tqdm(range(0, len(X), batch_size), desc=f"Epoch {epoch + 1}/{epochs}")
        for start in pbar:
            idx = perm[start : start + batch_size]
            x = Tensor(X[idx])
            optimizer.zero_grad()
            x_hat, mean, logvar = model(x)
            loss = vae_loss(x, x_hat, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
            seen = start + len(idx)
            pbar.set_postfix(loss=f"{total_loss / seen:.2f}")
    save_model(model, config, "vae.json")
    print("Saved model to vae.json")


if __name__ == "__main__":
    main()
