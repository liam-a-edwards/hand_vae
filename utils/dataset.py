from torchvision import datasets


def load(train):
    ds = datasets.EMNIST(root="data", split="balanced", train=train, download=True)
    x = ds.data.transpose(1, 2)  # Orientation
    x = (x.float() / 255 > 0.5).float()  # Binarize
    return x.reshape(len(x), -1), ds.targets


def load_data():
    X_train, y_train = load(True)
    X_test, y_test = load(False)
    return X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy()
