import json

import numpy as np

from utils.layers import VAE


def save_model(model, config, path):
    with open(path, "w") as f:
        json.dump(
            {"config": config, "params": [p.data.tolist() for p in model.parameters()]},
            f,
        )


def load_model(path):
    with open(path) as f:
        saved = json.load(f)
    model = VAE(**saved["config"])
    params = model.parameters()
    assert len(params) == len(saved["params"]), "Saved model does not match VAE"
    for p, value in zip(params, saved["params"]):
        value = np.array(value, dtype=float)
        assert value.shape == p.data.shape, (
            f"Shape mismatch: saved {value.shape}, model {p.data.shape}"
        )
        p.data = value

    return model
